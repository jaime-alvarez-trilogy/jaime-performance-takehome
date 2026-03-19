"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self._pending_consts = []  # buffered const loads, flushed in pairs by _flush_consts()

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        if engine != "load" and self._pending_consts:
            self._flush_consts()
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self._pending_consts.append(("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def _flush_consts(self):
        """Emit pending const loads in pairs (2 per bundle, using both load slots)."""
        pending = self._pending_consts
        self._pending_consts = []
        for i in range(0, len(pending), 2):
            ops = [pending[i]]
            if i + 1 < len(pending):
                ops.append(pending[i + 1])
            self.instrs.append({"load": ops})

    def _emit_hash_cycles(
        self, group_addrs,
        load_schedule=None,      # list of (load_op_list_or_None) per hash instruction
        nav_d_bounds_ops=None,   # pack into first mul or XOR even cycle with free slots
        nav_e_ops=None,          # pack into second such cycle
        gi_doubler_ops=None,     # pack into third available free cycle (after nav_d and nav_e)
        nv_fill_ops=None,        # Spec 04: pack nv fill ops into 4th free cycle
        flow_schedule=None,      # Spec 04 level-2: list of 9 FLOW ops (one per hash cycle, or None)
    ):
        """
        Core hash emission using multiply_add for additive stages.

        For stages with op1="+", op2="+", op3="<<":
          → 1 cycle: multiply_add(gv, gv, v_factor, v_C) for each group
        For XOR stages:
          → 2 cycles: odd (op1+op3 for all groups) + even (op2 for all groups)

        group_addrs: list of (gv, tmp1, tmp2) tuples.
        load_schedule: list of (load_ops_or_None) with one entry per emitted instruction.
          If None, no loads are emitted.
        nav_d_bounds_ops: deferred bounds-check ops; packed into the first available cycle
          with ≤ 3 valu ops (any multiply_add or XOR-even cycle).
        nav_e_ops: deferred multiply ops; packed into the second such cycle.
        gi_doubler_ops: gi = 2*gi + 1 ops for current triplet; packed into the third such cycle.
          These must be packed AFTER nav_d and nav_e to maintain proper dependency ordering.
        nv_fill_ops: Spec 04: nv fill ops for next triplet (multiply_add or vbroadcast);
          packed into the fourth available free cycle (after nav_d, nav_e, gi_doubler).
        """
        instrs_to_emit = []

        nav_d_packed = False
        nav_e_packed = False
        gi_doubler_packed = False
        nv_fill_packed = False

        def _try_pack_extra(valu_ops):
            nonlocal nav_d_packed, nav_e_packed, gi_doubler_packed, nv_fill_packed
            if not nav_d_packed and nav_d_bounds_ops and len(valu_ops) + len(nav_d_bounds_ops) <= SLOT_LIMITS["valu"]:
                valu_ops = valu_ops + nav_d_bounds_ops
                nav_d_packed = True
            elif not nav_e_packed and nav_e_ops and len(valu_ops) + len(nav_e_ops) <= SLOT_LIMITS["valu"]:
                valu_ops = valu_ops + nav_e_ops
                nav_e_packed = True
            elif not gi_doubler_packed and gi_doubler_ops and len(valu_ops) + len(gi_doubler_ops) <= SLOT_LIMITS["valu"]:
                valu_ops = valu_ops + gi_doubler_ops
                gi_doubler_packed = True
            elif not nv_fill_packed and nv_fill_ops and len(valu_ops) + len(nv_fill_ops) <= SLOT_LIMITS["valu"]:
                valu_ops = valu_ops + nv_fill_ops
                nv_fill_packed = True
            return valu_ops

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == "+" and op2 == "+" and op3 == "<<")

            if is_mul:
                # Single cycle: multiply_add for all groups (3 ops for full triplet)
                stage_info = self.hsc_new[hi]  # ("mul", v_factor, v_C)
                _, v_factor, v_C = stage_info
                valu_ops = [("multiply_add", gv, gv, v_factor, v_C) for gv, tmp1, tmp2 in group_addrs]
                valu_ops = _try_pack_extra(valu_ops)
                instrs_to_emit.append(valu_ops)
            else:
                # Two cycles: XOR stage
                stage_info = self.hsc_new[hi]  # ("xor", op1, v_C, op2, op3, v_shift)
                _, op1_s, v_c1, op2_s, op3_s, v_c2 = stage_info

                # Odd cycle: op1+op3 for all groups (6 ops for 3-group triplet)
                valu_c1 = []
                for gv, tmp1, tmp2 in group_addrs:
                    valu_c1.append((op1_s, tmp1, gv, v_c1))
                    valu_c1.append((op3_s, tmp2, gv, v_c2))
                instrs_to_emit.append(valu_c1)

                # Even cycle: op2 for all groups (3 ops for 3-group triplet), 3 free slots
                valu_c2 = [(op2_s, gv, tmp1, tmp2) for gv, tmp1, tmp2 in group_addrs]
                valu_c2 = _try_pack_extra(valu_c2)
                instrs_to_emit.append(valu_c2)

        # Emit with load schedule and optional FLOW schedule
        for idx, valu_ops in enumerate(instrs_to_emit):
            bundle = {"valu": valu_ops}
            if load_schedule and idx < len(load_schedule) and load_schedule[idx]:
                bundle["load"] = load_schedule[idx]
            if flow_schedule and idx < len(flow_schedule) and flow_schedule[idx] is not None:
                bundle["flow"] = [flow_schedule[idx]]
            self.instrs.append(bundle)

    def emit_triplet_hash(self, group_addrs):
        """
        Emit interleaved VALU hash for 2–3 independent groups (no prefetch loads).
        Uses multiply_add for additive stages (stages 0, 2, 4): saves 3 cycles per triplet.
        """
        self._emit_hash_cycles(group_addrs)

    def _make_load_schedule(self, total_loads, nv_next_base, addr_next_base, n_hash_cycles):
        """
        Build a load schedule: a list of (load_ops_or_None) with one entry per hash instruction.
        Spreads total_loads load_offset ops across hash instructions, 2 per instruction.
        """
        schedule = []
        load_idx = 0
        for i in range(n_hash_cycles):
            if load_idx < total_loads:
                g_idx = load_idx // VLEN
                vi = load_idx % VLEN
                schedule.append([
                    ("load_offset", nv_next_base + g_idx * VLEN,
                     addr_next_base + g_idx * VLEN, vi),
                    ("load_offset", nv_next_base + g_idx * VLEN,
                     addr_next_base + g_idx * VLEN, vi + 1),
                ])
                load_idx += 2
            else:
                schedule.append(None)
        return schedule

    def _count_hash_cycles(self):
        """Count number of instruction bundles emitted by _emit_hash_cycles (no nav ops)."""
        count = 0
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == "+" and op2 == "+" and op3 == "<<")
            count += 1 if is_mul else 2
        return count

    def _emit_triplet_hash_core(
        self, group_addrs, nv_next_base, addr_next_base, n_next_groups,
        nav_d_bounds_ops=None, nav_e_ops=None, gi_doubler_ops=None,
        nv_fill_ops=None, flow_schedule=None, preloaded_loads=0
    ):
        """
        Common core for hash emission with optional deferred nav ops and prefetch loads.
        Returns extra_load_pairs (list of load pairs that didn't fit in hash cycles).
        preloaded_loads: number of loads already pre-emitted (reduces extra_load_pairs).
        """
        total_loads = n_next_groups * VLEN - preloaded_loads
        n_hash_cycles = self._count_hash_cycles()
        max_hash_loads = n_hash_cycles * 2
        load_schedule = self._make_load_schedule(
            min(total_loads, max_hash_loads), nv_next_base, addr_next_base, n_hash_cycles
        ) if n_next_groups > 0 else None
        self._emit_hash_cycles(
            group_addrs, load_schedule=load_schedule,
            nav_d_bounds_ops=nav_d_bounds_ops, nav_e_ops=nav_e_ops,
            gi_doubler_ops=gi_doubler_ops, nv_fill_ops=nv_fill_ops,
            flow_schedule=flow_schedule
        )
        # Compute leftover loads that didn't fit in hash
        extra_load_pairs = []
        for load_idx in range(max_hash_loads, total_loads, 2):
            g_idx = load_idx // VLEN
            vi = load_idx % VLEN
            extra_load_pairs.append([
                ("load_offset", nv_next_base + g_idx * VLEN,
                 addr_next_base + g_idx * VLEN, vi),
                ("load_offset", nv_next_base + g_idx * VLEN,
                 addr_next_base + g_idx * VLEN, vi + 1),
            ])
        return extra_load_pairs

    def emit_triplet_hash(self, group_addrs, gi_doubler_ops=None, nv_fill_ops=None,
                          flow_schedule=None):
        """
        Emit interleaved VALU hash for 2–3 independent groups (no prefetch loads).
        Uses multiply_add for additive stages (stages 0, 2, 4): saves 3 cycles per triplet.
        gi_doubler_ops: optional ops to pack into a free cycle (gi = 2*gi+1 for current groups).
        nv_fill_ops: Spec 04: optional nv fill ops packed into a free cycle.
        flow_schedule: Spec 04 level-2: optional list of 9 FLOW ops for nv vselect.
        """
        self._emit_triplet_hash_core(group_addrs, 0, 0, 0, gi_doubler_ops=gi_doubler_ops,
                                     nv_fill_ops=nv_fill_ops, flow_schedule=flow_schedule)

    def emit_triplet_hash_with_prefetch(
        self, group_addrs, nv_next_base, addr_next_base, n_next_groups,
        gi_doubler_ops=None
    ):
        """
        Emit interleaved hash for 2-3 groups with scatter-load prefetch in idle load slots.
        Uses multiply_add for additive stages (9 cycles total for full triplet).

        With 9 hash cycles and 24 loads needed (n_next=3), only 18 loads fit.
        The remaining 6 loads must be added to nav cycles by the caller.
        For n_next ≤ 2 (16 loads): all fit in 9 cycles × 2 = 18 load slots.

        addr_next must already be written before this is called.

        gi_doubler_ops: optional multiply_add ops (gi = 2*gi+1) packed in a free hash cycle.

        Returns a list of extra load_offset ops that didn't fit in hash cycles.
        Each entry is a list of 2 load ops to append to a subsequent nav bundle.
        """
        return self._emit_triplet_hash_core(
            group_addrs, nv_next_base, addr_next_base, n_next_groups,
            gi_doubler_ops=gi_doubler_ops
        )

    def emit_triplet_hash_with_prefetch_and_nav_tail(
        self, group_addrs, nv_next_base, addr_next_base, n_next_groups,
        nav_d_bounds_ops, nav_e_ops, gi_doubler_ops=None, nv_fill_ops=None,
        flow_schedule=None, preloaded_loads=0
    ):
        """
        Emit interleaved hash for 2-3 groups with scatter-load prefetch AND deferred nav ops
        from prior triplet packed into free valu slots.

        Uses multiply_add for additive stages (9 cycles total).
        nav_d_bounds_ops and nav_e_ops are packed into the first two cycles that have free
        valu slots (multiply_add cycles or XOR even cycles, all have 3 free slots for 3-group).

        gi_doubler_ops: optional multiply_add ops (gi = 2*gi+1) packed in the third free cycle.
        nv_fill_ops: Spec 04: optional nv fill ops packed into the fourth free cycle.

        Dependencies:
          - nav_d_bounds reads gi from nav-B (before this hash begins) ✓
          - nav_e reads tvld from nav_d_bounds (packed in cycle ≥ hash cycle 1;
            nav_e packed in cycle ≥ hash cycle 2, at least 2 cycles later) ✓
          - gi_doubler reads gi of CURRENT triplet (different registers from prev nav);
            packed after nav_d and nav_e, reads gi which was last written by nav_e of
            the current triplet from PREVIOUS round (committed long before this hash) ✓
          - addr_next read by prefetch loads was set before this call ✓

        With 9 hash cycles and n_next=3 (24 loads needed): only 18 load slots available.
        The remaining 6 loads must go in nav cycles (added by caller after this method).

        Returns a list of extra load_offset op pairs that didn't fit in hash cycles.
        Each entry is a list of 2 load ops to append to a subsequent nav bundle.
        """
        return self._emit_triplet_hash_core(
            group_addrs, nv_next_base, addr_next_base, n_next_groups,
            nav_d_bounds_ops=nav_d_bounds_ops, nav_e_ops=nav_e_ops,
            gi_doubler_ops=gi_doubler_ops, nv_fill_ops=nv_fill_ops,
            flow_schedule=flow_schedule, preloaded_loads=preloaded_loads
        )

    def _emit_alu_hash_stages(self, alu_group_addr, stage_start=0, stage_end=6):
        """
        Generate ALU ops for one group's 6-stage hash pipeline using scratch addresses.
        Returns a list of cycles (each cycle is a list of (op, dest, src1, src2) tuples).

        Total: 15 cycles for stages 0-5:
          - Mul stages (0, 2, 4): 2 cycles each (8 multiply + 8 add = 16 ops)
          - XOR stages (1, 3, 5): 3 cycles each (8 op1 + 8 op3 + 8 op2 = 24 ops)

        alu_group_addr: (a_base, tmp1_base, tmp2_base) scratch addresses for VLEN lanes.
        The ALU hash computes the same result as the VALU hash on each lane independently.
        """
        a_base, tmp1_base, tmp2_base = alu_group_addr
        cycles = []
        for hi in range(stage_start, stage_end):
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            is_mul = (op1 == "+" and op2 == "+" and op3 == "<<")
            if is_mul:
                factor = (1 + (1 << val3)) & 0xFFFFFFFF
                c_factor = self.const_map[factor]
                c_C = self.const_map[val1]
                # Cycle 1: a = a * factor (= a * (1+2^val3))
                cycles.append([("*", a_base + i, a_base + i, c_factor) for i in range(VLEN)])
                # Cycle 2: a = a + C (= a + val1)
                cycles.append([("+", a_base + i, a_base + i, c_C) for i in range(VLEN)])
            else:
                c_val1 = self.const_map[val1]
                c_val3 = self.const_map[val3]
                # Cycle 1: tmp1 = a op1 val1  (reads old a)
                cycles.append([(op1, tmp1_base + i, a_base + i, c_val1) for i in range(VLEN)])
                # Cycle 2: tmp2 = a op3 val3  (reads old a — not tmp1, preserving myhash semantics)
                cycles.append([(op3, tmp2_base + i, a_base + i, c_val3) for i in range(VLEN)])
                # Cycle 3: a = tmp1 op2 tmp2
                cycles.append([(op2, a_base + i, tmp1_base + i, tmp2_base + i) for i in range(VLEN)])
        return cycles

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Phase 1: SIMD + VLIW optimized kernel.

        Key ideas:
          - SIMD: process VLEN=8 inputs per cycle using VALU instructions
          - VLIW: pack hash op1+op3 into same cycle (they're independent)
          - Scratch-resident state: all 256 idx/val live in scratch across all
            rounds, eliminating 8192 memory read/writes from the baseline
          - Scatter loads: 8 node_val loads per group (4 cycles at 2/cycle)

        Per SIMD group (8 inputs):
          1 cycle  - addr[i] = forest_ptr + idx[i]   (valu add)
          4 cycles - scatter load node_val[0..7]      (2 loads/cycle)
          1 cycle  - val ^= node_val                  (valu xor)
         12 cycles - hash: 6 stages x 2 cycles each  (op1+op3 packed)
          5 cycles - navigation                       (branch + wrap)
        = 23 cycles/group x 512 groups ~ 11,776 cycles
        """
        n_groups = batch_size // VLEN  # 32 groups of 8

        # ── Scratch allocation ────────────────────────────────────────────────
        # 3-group interleaved hash: separate tmp regions per concurrent group
        self.tmp1_A = self.alloc_scratch("tmp1_A", VLEN)
        self.tmp1_B = self.alloc_scratch("tmp1_B", VLEN)
        self.tmp1_C = self.alloc_scratch("tmp1_C", VLEN)
        self.tmp2_A = self.alloc_scratch("tmp2_A", VLEN)
        self.tmp2_B = self.alloc_scratch("tmp2_B", VLEN)
        self.tmp2_C = self.alloc_scratch("tmp2_C", VLEN)
        self.nv_A      = self.alloc_scratch("nv_A",      VLEN * 3)  # node values, buffer A (3 groups)
        self.nv_B      = self.alloc_scratch("nv_B",      VLEN * 3)  # node values, buffer B (ping-pong)
        self.addr_next = self.alloc_scratch("addr_next", VLEN * 3)  # scatter addrs for next triplet
        # Spec 03 FR2: per-group nav temporaries (packed multi-group navigation)
        tmp_nav_A = self.alloc_scratch("tmp_nav_A", VLEN)
        tmp_nav_B = self.alloc_scratch("tmp_nav_B", VLEN)
        tmp_nav_C = self.alloc_scratch("tmp_nav_C", VLEN)
        tmp_vld_A = self.alloc_scratch("tmp_vld_A", VLEN)
        tmp_vld_B = self.alloc_scratch("tmp_vld_B", VLEN)
        tmp_vld_C = self.alloc_scratch("tmp_vld_C", VLEN)
        tmp_s    = self.alloc_scratch("tmp_s")            # scalar scratch
        nv_root_scalar = self.alloc_scratch("nv_root_scalar", 1)  # forest_values[0] (all gi=0 at round 0)
        # Spec 04 FR1: scratch for software nv constants (rounds 1, 12 where gi ∈ {1,2})
        nv1_scalar = self.alloc_scratch("nv1_scalar", 1)  # forest_values[1]
        nv2_scalar = self.alloc_scratch("nv2_scalar", 1)  # forest_values[2]
        nv_delta_scalar = self.alloc_scratch("nv_delta_scalar", 1)  # (nv2 - nv1) % 2^32
        nv_base_scalar  = self.alloc_scratch("nv_base_scalar", 1)   # (nv1 - delta) % 2^32
        v_nv_delta = self.alloc_scratch("v_nv_delta", VLEN)  # broadcast of nv_delta_scalar
        v_nv_base  = self.alloc_scratch("v_nv_base",  VLEN)  # broadcast of nv_base_scalar
        # Spec 04 level-2: scratch for software nv (rounds 2, 13 where gi ∈ {3..6})
        nv3_scalar = self.alloc_scratch("nv3_scalar", 1)  # forest_values[3]
        nv4_scalar = self.alloc_scratch("nv4_scalar", 1)  # forest_values[4]
        nv5_scalar = self.alloc_scratch("nv5_scalar", 1)  # forest_values[5]
        nv6_scalar = self.alloc_scratch("nv6_scalar", 1)  # forest_values[6]
        v_nv3 = self.alloc_scratch("v_nv3", VLEN)
        v_nv4 = self.alloc_scratch("v_nv4", VLEN)
        v_nv5 = self.alloc_scratch("v_nv5", VLEN)
        v_nv6 = self.alloc_scratch("v_nv6", VLEN)
        v_five = self.alloc_scratch("v_five", VLEN)  # vbroadcast(5) for mask_A = (gi < 5)
        # Temp buffers for 3-group FLOW-based nv computation (shared across triplets)
        mask_A_bufs = [self.alloc_scratch(f"mask_A_{i}", VLEN) for i in range(3)]
        nv_lo_bufs  = [self.alloc_scratch(f"nv_lo_{i}",  VLEN) for i in range(3)]
        nv_hi_bufs  = [self.alloc_scratch(f"nv_hi_{i}",  VLEN) for i in range(3)]
        mask_B_bufs = [self.alloc_scratch(f"mask_B_{i}", VLEN) for i in range(3)]

        # Memory header pointers (loaded from mem[4..6]) — pack 2 per instruction
        self.alloc_scratch("forest_values_p")
        self.alloc_scratch("inp_indices_p")
        self.alloc_scratch("inp_values_p")
        # Pre-load const addresses for header indices 4, 5, 6, then flush
        c4 = self.scratch_const(4)
        c5 = self.scratch_const(5)
        c6 = self.scratch_const(6)
        self._flush_consts()
        # Instruction: forest_values_p + inp_indices_p
        self.instrs.append({"load": [
            ("load", self.scratch["forest_values_p"], self.scratch_const(4)),
            ("load", self.scratch["inp_indices_p"],   self.scratch_const(5)),
        ]})
        # Instruction: inp_values_p + nv_root_scalar (forest_values[0] = root node value)
        # All inputs start at gi=0, so nv for round 0 = forest_values[0] for all groups.
        # forest_values_p is ready (written at end of previous instruction). ✓
        self.instrs.append({"load": [
            ("load", self.scratch["inp_values_p"], self.scratch_const(6)),
            ("load", nv_root_scalar, self.scratch["forest_values_p"]),
        ]})

        # ── Spec 04: Load forest_values[1..6] for software nv ───────────────
        # Allocate ALL fp+X scratch vars upfront so we can emit one combined ALU cycle.
        fp_plus1 = self.alloc_scratch("fp_plus1", 1)
        fp_plus2 = self.alloc_scratch("fp_plus2", 1)
        fp_plus3 = self.alloc_scratch("fp_plus3", 1)
        fp_plus4 = self.alloc_scratch("fp_plus4", 1)
        fp_plus5 = self.alloc_scratch("fp_plus5", 1)
        fp_plus6 = self.alloc_scratch("fp_plus6", 1)
        c1_fp = self.scratch_const(1)
        c2_fp = self.scratch_const(2)
        # Flush c1, c2 together → {load: [const(1), const(2)]}
        self._flush_consts()
        # c3 is a new const; c4/c5/c6 already flushed in header section above.
        c3_fp = self.scratch_const(3)
        c4_fp = self.scratch_const(4)   # reuses header-flushed const
        c5_fp = self.scratch_const(5)   # reuses header-flushed const
        c6_fp = self.scratch_const(6)   # reuses header-flushed const
        # Combine c3 flush with ALU fp+1,fp+2 (independent: fp+1,fp+2 use c1,c2 not c3).
        # This merges two operations into one cycle via VLIW load+ALU parallelism.
        assert len(self._pending_consts) == 1, "expected only const(3) pending"
        c3_flush_op = self._pending_consts.pop(0)
        self._pending_consts = []
        # Cycle A: load const(3) [load slot] + compute fp+1, fp+2 [ALU slots]
        self.instrs.append({
            "load": [c3_flush_op],
            "alu": [
                ("+", fp_plus1, self.scratch["forest_values_p"], c1_fp),
                ("+", fp_plus2, self.scratch["forest_values_p"], c2_fp),
            ]
        })
        # Cycle B: load nv1, nv2 + compute fp+3..6 (c3 ready from A; fp+1,fp+2 from A)
        self.instrs.append({
            "load": [("load", nv1_scalar, fp_plus1), ("load", nv2_scalar, fp_plus2)],
            "alu": [
                ("+", fp_plus3, self.scratch["forest_values_p"], c3_fp),
                ("+", fp_plus4, self.scratch["forest_values_p"], c4_fp),
                ("+", fp_plus5, self.scratch["forest_values_p"], c5_fp),
                ("+", fp_plus6, self.scratch["forest_values_p"], c6_fp),
            ]
        })
        # Cycle C: load nv3, nv4 + compute delta = nv2 - nv1 (fp+3,4 from B; nv1,2 from B)
        self.instrs.append({
            "load": [("load", nv3_scalar, fp_plus3), ("load", nv4_scalar, fp_plus4)],
            "alu": [("-", nv_delta_scalar, nv2_scalar, nv1_scalar)]
        })
        # Cycle D: load nv5, nv6 + compute base = nv1 - delta (fp+5,6 from B; delta from C)
        self.instrs.append({
            "load": [("load", nv5_scalar, fp_plus5), ("load", nv6_scalar, fp_plus6)],
            "alu": [("-", nv_base_scalar, nv1_scalar, nv_delta_scalar)]
        })

        # ── Scalar constants ──────────────────────────────────────────────────
        c0 = self.scratch_const(0, "c0")
        c1 = self.scratch_const(1, "c1")
        c2 = self.scratch_const(2, "c2")
        c_nn = self.scratch_const(n_nodes, "c_nnodes")

        # ── Vector constants (broadcast once, reused every cycle) ─────────────
        v_one    = self.alloc_scratch("v_one",    VLEN)
        v_two    = self.alloc_scratch("v_two",    VLEN)  # for gi-doubler: gi = 2*gi + 1
        v_nnodes = self.alloc_scratch("v_nnodes", VLEN)
        v_fp     = self.alloc_scratch("v_fp",     VLEN)  # forest_values_p as vector

        # Hash stage vector constants
        # For stages using multiply_add: factor = 1 + 2^shift (for "+", C, "+", "<<", shift)
        # Stages 0, 2, 4 have op1="+", op2="+", op3="<<": use multiply_add(dest, gv, v_factor, v_C)
        # Stages 1, 3, 5 have XOR ops: use standard 2-cycle interleave
        hvc = {}  # raw value -> vector scratch addr (for XOR stage constants and C values)

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # multiply_add stage: gv = gv * (1 + 2^val3) + val1
                # Need v_C (=val1) and v_factor (= 1 + 2^val3)
                factor = (1 + (1 << val3)) % (2**32)
                for val in (val1, factor):
                    if val not in hvc:
                        self.scratch_const(val)  # adds to pending_consts
                        cv = self.alloc_scratch(f"hc_{val:#x}", VLEN)
                        hvc[val] = cv
            else:
                # XOR stage: standard op1(tmp1, gv, C) + op3(tmp2, gv, shift); op2(gv, tmp1, tmp2)
                for val in (val1, val3):
                    if val not in hvc:
                        self.scratch_const(val)  # adds to pending_consts
                        cv = self.alloc_scratch(f"hc_{val:#x}", VLEN)
                        hvc[val] = cv

        # Build hash stage constants list for use in hash emission
        # hsc[hi] = info for stage hi:
        #   For multiply_add: ("mul", v_factor, v_C)
        #   For XOR: ("xor", v_C, v_shift)
        hsc_new = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = (1 + (1 << val3)) % (2**32)
                hsc_new.append(("mul", hvc[factor], hvc[val1]))
            else:
                hsc_new.append(("xor", op1, hvc[val1], op2, op3, hvc[val3]))
        self.hsc_new = hsc_new

        # self.hsc is no longer used (all hash methods use hsc_new)
        self.hsc = None

        # Flush all pending scalar consts before vbroadcast (which reads them)
        self._flush_consts()

        # Pack all vbroadcast ops (up to 6 per instruction using 6 valu slots).
        # Includes nv_A initialization: all inputs start at gi=0, so all groups need
        # forest_values[0] (already in nv_root_scalar). Replaces the 13-cycle round-0 prologue.
        all_broadcasts = [
            ("vbroadcast", v_one,    c1),
            ("vbroadcast", v_two,    c2),
            ("vbroadcast", v_nnodes, c_nn),
            ("vbroadcast", v_fp,     self.scratch["forest_values_p"]),
            # nv_A for 3 groups: broadcast forest_values[0] to all 24 lanes
            ("vbroadcast", self.nv_A,             nv_root_scalar),
            ("vbroadcast", self.nv_A + VLEN,      nv_root_scalar),
            ("vbroadcast", self.nv_A + 2 * VLEN,  nv_root_scalar),
            # Spec 04 FR1: software nv constants for rounds with gi ∈ {1,2}
            ("vbroadcast", v_nv_delta, nv_delta_scalar),
            ("vbroadcast", v_nv_base,  nv_base_scalar),
            # Spec 04 level-2: software nv constants for rounds with gi ∈ {3..6}
            ("vbroadcast", v_nv3,  nv3_scalar),
            ("vbroadcast", v_nv4,  nv4_scalar),
            ("vbroadcast", v_nv5,  nv5_scalar),
            ("vbroadcast", v_nv6,  nv6_scalar),
            ("vbroadcast", v_five, self.scratch_const(5)),
        ] + [
            ("vbroadcast", hvc[val], self.const_map[val])
            for val in hvc
        ]
        # Deduplicate (same val may appear multiple times)
        seen_cv = set()
        unique_broadcasts = []
        for op in all_broadcasts:
            if op[1] not in seen_cv:
                seen_cv.add(op[1])
                unique_broadcasts.append(op)
        VALU_LIMIT = SLOT_LIMITS["valu"]

        # ── All 256 idx/val arrays resident in scratch ────────────────────────
        # Eliminates per-round loads/stores — only loaded once, stored once.
        # addr_arr: scratch buffer for precomputed scatter addresses (reused for idx and val)
        addr_arr = self.alloc_scratch("addr_arr", n_groups)  # 32 scalar addrs
        idx_base = self.alloc_scratch("idx_arr", VLEN * n_groups)  # 256 words
        val_base = self.alloc_scratch("val_arr", VLEN * n_groups)  # 256 words

        # Stride-based addressing optimization:
        # Instead of loading 30 g*VLEN consts (1 per group), only load consts for g=0..11
        # (first ALU batch). ALU batches 1,2 compute addr_arr[g] from addr_arr[g-12] + c_stride
        # where c_stride = 12*VLEN = 96 (for batch 1) or 24*VLEN = 192 (for batch 2).
        # This reduces const loading from 15 cycles to 5+1 = 6 cycles (10 new consts for g=1..11
        # plus c96 and c192 paired in one load cycle).
        ALU_LIMIT = SLOT_LIMITS["alu"]
        stride1 = ALU_LIMIT * VLEN       # 96 = 12 * 8
        stride2 = 2 * ALU_LIMIT * VLEN  # 192 = 24 * 8

        # Allocate consts for g=0..11 (only new ones; g=0→0 and g=2→16 already in const_map)
        # and for strides c96, c192.
        small_gvlen_ops = []  # NEW consts for g=0..11
        for g in range(min(ALU_LIMIT, n_groups)):
            v = g * VLEN
            if v not in self.const_map:
                addr_c = self.alloc_scratch()
                small_gvlen_ops.append(("const", addr_c, v))
                self.const_map[v] = addr_c

        # Allocate stride constants c96 and c192
        c_stride1 = self.scratch_const(stride1)  # → queued in _pending_consts
        c_stride2 = self.scratch_const(stride2)  # → queued in _pending_consts

        # Emit vbroadcast batches overlapped with small_gvlen const load cycles.
        # Vbroadcasts use VALU engine; consts use LOAD engine → independent.
        # Group small_gvlen_ops + pending stride consts into pairs:
        all_early_consts = small_gvlen_ops + list(self._pending_consts)
        self._pending_consts = []  # consumed above

        early_const_pairs = []
        for i in range(0, len(all_early_consts), 2):
            pair = [all_early_consts[i]]
            if i + 1 < len(all_early_consts):
                pair.append(all_early_consts[i + 1])
            early_const_pairs.append(pair)

        # Emit vbroadcast batches overlapped with early const pairs
        vb_batches = []
        for batch_start in range(0, len(unique_broadcasts), VALU_LIMIT):
            vb_batches.append(unique_broadcasts[batch_start:batch_start + VALU_LIMIT])

        max_cycles = max(len(vb_batches), len(early_const_pairs))
        for i in range(max_cycles):
            bundle = {}
            if i < len(vb_batches):
                bundle["valu"] = list(vb_batches[i])
            if i < len(early_const_pairs):
                bundle["load"] = list(early_const_pairs[i])
            self.instrs.append(bundle)

        # ALU addressing with stride-based batches:
        # Batch 0: addr_arr[g] = base_ptr + g*VLEN  (g=0..ALU_LIMIT-1, direct const offsets)
        # Batch 1: addr_arr[g] = addr_arr[g-ALU_LIMIT] + c_stride1  (g=ALU_LIMIT..2*ALU_LIMIT-1)
        # Batch 2: addr_arr[g] = addr_arr[g-2*ALU_LIMIT] + c_stride2 (g=2*ALU_LIMIT..n_groups-1)
        # All 3 batches can run consecutively: batch 0 produces addr_arr[0..11],
        # batch 1 reads addr_arr[0..11] (written by batch 0), adds stride1.
        # Batch 2 reads addr_arr[0..7] (written by batch 0), adds stride2.
        # Vload pair 0 overlaps with batch 1 (reads addr_arr[0,1] from batch 0 at START of batch 1).
        # Vload pair 1 overlaps with batch 2 (reads addr_arr[2,3] from batch 0 at START of batch 2).

        def make_idx_alu_batches(base_ptr):
            b0 = [("+", addr_arr + g, base_ptr, self.const_map[g * VLEN])
                  for g in range(min(ALU_LIMIT, n_groups))]
            b1 = [("+", addr_arr + g, addr_arr + (g - ALU_LIMIT), c_stride1)
                  for g in range(ALU_LIMIT, min(2 * ALU_LIMIT, n_groups))]
            b2 = [("+", addr_arr + g, addr_arr + (g - 2 * ALU_LIMIT), c_stride2)
                  for g in range(2 * ALU_LIMIT, n_groups)]
            return [b0, b1, b2] if b2 else ([b0, b1] if b1 else [b0])

        # Build vload pairs for idx and val
        def make_vload_pairs(base):
            pairs = []
            for g in range(0, n_groups, 2):
                ops = [("vload", base + g * VLEN, addr_arr + g)]
                if g + 1 < n_groups:
                    ops.append(("vload", base + (g+1) * VLEN, addr_arr + g + 1))
                pairs.append(ops)
            return pairs

        val_vload_pairs = make_vload_pairs(val_base)

        # ── Idx initialization: all inputs start at gi=0, so zero-initialize idx in scratch.
        # Submission tests only check val (not idx), so we don't need to load from memory.
        # Use vbroadcast(0) to set all idx lanes to 0. Overlapped with val loading below.
        idx_vbroadcasts = [
            ("vbroadcast", idx_base + g * VLEN, c0)
            for g in range(n_groups)
        ]
        # Group into batches of VALU_LIMIT
        idx_vb_batches = []
        for i in range(0, len(idx_vbroadcasts), VALU_LIMIT):
            idx_vb_batches.append(idx_vbroadcasts[i:i + VALU_LIMIT])

        # Val phase: compute addr_arr for val addresses, then vload.
        val_alu_batches = make_idx_alu_batches(self.scratch["inp_values_p"])
        # Emit val ALU batch 0 (pure ALU, no overlap)
        bundle0 = {"alu": val_alu_batches[0]}
        if idx_vb_batches:
            bundle0["valu"] = idx_vb_batches[0]
        self.instrs.append(bundle0)

        # Emit val ALU batches 1,2 overlapped with vload pairs and idx vbroadcasts
        vb_idx = 1  # next idx_vb_batch to use
        vl_idx = 0  # next val_vload_pair to use
        for alu_batch_i in range(1, len(val_alu_batches)):
            bundle = {"alu": val_alu_batches[alu_batch_i]}
            if vl_idx < len(val_vload_pairs):
                bundle["load"] = val_vload_pairs[vl_idx]
                vl_idx += 1
            if vb_idx < len(idx_vb_batches):
                bundle["valu"] = idx_vb_batches[vb_idx]
                vb_idx += 1
            self.instrs.append(bundle)

        # Remaining vload pairs overlapped with idx vbroadcasts
        while vl_idx < len(val_vload_pairs) or vb_idx < len(idx_vb_batches):
            bundle = {}
            if vl_idx < len(val_vload_pairs):
                bundle["load"] = val_vload_pairs[vl_idx]
                vl_idx += 1
            if vb_idx < len(idx_vb_batches):
                bundle["valu"] = idx_vb_batches[vb_idx]
                vb_idx += 1
            if bundle:
                self.instrs.append(bundle)

        self.add("flow", ("pause",))

        # ── Main loop (unrolled: rounds x triplets of groups) ────────────────
        # Spec 01: 3-group interleaved hash pipeline.
        # Spec 02: load/compute overlap with double-buffered nv_A/nv_B.
        # Spec 03: packed multi-group navigation + vselect→multiply + xor overlap.
        tmp1_slots = [self.tmp1_A, self.tmp1_B, self.tmp1_C]
        tmp2_slots = [self.tmp2_A, self.tmp2_B, self.tmp2_C]
        # Per-group nav temps: index into these by position within triplet (0,1,2)
        tmp_nav_slots = [tmp_nav_A, tmp_nav_B, tmp_nav_C]
        tmp_vld_slots = [tmp_vld_A, tmp_vld_B, tmp_vld_C]
        n_triplets = (n_groups + 2) // 3  # ceil(n_groups / 3) = 11 for n_groups=32
        nv_bufs = [self.nv_A, self.nv_B]

        first_groups = [g for g in [0, 1, 2] if g < n_groups]

        # Cross-round deferred nav-D/E: carry prev_nav_d_bounds and prev_nav_e across rounds.
        # For non-last rounds, ti=10's nav-D and nav-E are deferred into round r+1's ti=0 hash.
        prev_nav_d_bounds = []
        # Spec 04: Flag to skip ti=0 xor if it was pre-applied by previous round's nav.
        skip_ti0_xor = False
        prev_nav_e = []

        # Store slot limit per instruction bundle
        STORE_LIMIT = SLOT_LIMITS["store"]  # 2

        def inject_stores(bundle, pending):
            """Add up to STORE_LIMIT store ops from pending to bundle. Modifies bundle in-place."""
            if pending:
                take = min(STORE_LIMIT, len(pending))
                bundle["store"] = [pending.pop(0) for _ in range(take)]

        for _r in range(rounds):
            # In the last round: queue store ops after each triplet's hash completes.
            # Stores use the STORE engine (independent from VALU/LOAD/ALU), so they can
            # be added to any nav bundle without affecting other engines.
            is_last_round = (_r == rounds - 1)
            pending_stores = []  # list of ('vstore', addr_arr+g, val_base+g*VLEN) ops

            # ── Spec 04: Round classification ─────────────────────────────────
            # Classify each round to enable per-round-class optimizations.
            is_level1_round = (_r == 1 or _r == rounds - 4)   # rounds 1, 12: gi ∈ {1,2}
            is_root_round   = (_r == 0 or _r == 11)            # rounds 0, 11: gi=0
            is_wrap_round   = (_r == rounds - 6)               # round 10: guaranteed all-wrap
            is_level2_round = (_r == 2 or _r == rounds - 3)   # rounds 2, 13: gi ∈ {3..6}
            # Determine if next round uses software nv (so we skip preloading in last triplet)
            next_r = _r + 1
            next_is_software_nv = (
                next_r < rounds and
                (next_r == 1 or next_r == rounds - 4 or next_r == 11
                 or next_r == 2 or next_r == rounds - 3)
            )

            # ── Spec 04: nv_A for software_nv rounds was pre-filled by the
            # previous round's last triplet hash (nv_fill_ops_for_hash). No
            # standalone prologue cycle needed. ─────────────────────────────

            # ── Main triplet loop ─────────────────────────────────────────────
            # nav-D bounds and nav-E of triplet ti are deferred into triplet ti+1's hash
            # even cycles (hi=0 and hi=1), saving 2 cycles per non-last triplet.
            # For non-last rounds, ti=10's nav-D/E are deferred into the NEXT ROUND's
            # ti=0 hash (cross-round deferred nav), saving 2 cycles per round.
            # prev_nav_d_bounds and prev_nav_e are carried across round boundaries.

            nv_preloaded_early = False  # set True at ti=n_triplets-2 when vi=6,7 of group 2 pre-loaded
            for ti in range(n_triplets):
                a, b, c = ti * 3, ti * 3 + 1, ti * 3 + 2
                groups = [g for g in [a, b, c] if g < n_groups]
                nv_cur      = nv_bufs[ti % 2]
                nv_next_buf = nv_bufs[(ti + 1) % 2]
                is_last = (ti == n_triplets - 1)

                # Next triplet's groups (for addr_next and prefetch)
                if not is_last:
                    na, nb, nc = (ti + 1) * 3, (ti + 1) * 3 + 1, (ti + 1) * 3 + 2
                    next_groups = [g for g in [na, nb, nc] if g < n_groups]
                else:
                    next_groups = []

                # Spec 04: For level1/root/level2 rounds, no scatter loads needed for nv.
                # Instead, nv is computed via multiply_add (level1), vbroadcast (root),
                # or FLOW vselect (level2).
                software_nv_round = is_level1_round or is_root_round or is_level2_round
                # Effective n_next_groups for scatter load purposes
                scatter_n_next = 0 if software_nv_round else len(next_groups)

                # Spec 04: Build nv_fill_ops and flow_schedule for next triplet
                # (nv_fill_ops packed into hash free VALU slots; flow_schedule uses FLOW slots)
                nv_fill_ops_for_hash = None
                flow_schedule_for_hash = None
                if software_nv_round and not is_last:
                    # Within software_nv round: fill nv for next within-round triplet
                    if is_level1_round:
                        nv_fill_ops_for_hash = [
                            ("multiply_add", nv_next_buf + j * VLEN,
                             idx_base + ng * VLEN, v_nv_delta, v_nv_base)
                            for j, ng in enumerate(next_groups)
                        ]
                    elif is_root_round:
                        nv_fill_ops_for_hash = [
                            ("vbroadcast", nv_next_buf + j * VLEN, nv_root_scalar)
                            for j in range(len(next_groups))
                        ]
                    elif is_level2_round:
                        # Level-2: mask_B (gi & 1) as nv_fill VALU; FLOW vselect for nv_lo/hi/final
                        n_ng = len(next_groups)
                        nv_fill_ops_for_hash = [
                            ("&", mask_B_bufs[j], idx_base + ng * VLEN, v_one)
                            for j, ng in enumerate(next_groups)
                        ]
                        # FLOW schedule: 9 ops across 9 hash cycles
                        # mask_A for next_groups pre-computed (in xor_bundle or prev nav-A)
                        # nv_fill (mask_B) packed into cycle 5 (4th free VALU slot)
                        flow_schedule_for_hash = [None] * 9
                        for j in range(n_ng):
                            flow_schedule_for_hash[j * 2]     = ("vselect", nv_lo_bufs[j], mask_A_bufs[j], v_nv3, v_nv5)
                            flow_schedule_for_hash[j * 2 + 1] = ("vselect", nv_hi_bufs[j], mask_A_bufs[j], v_nv4, v_nv6)
                        for j in range(n_ng):
                            flow_schedule_for_hash[6 + j] = ("vselect", nv_next_buf + j * VLEN, mask_B_bufs[j], nv_lo_bufs[j], nv_hi_bufs[j])
                elif is_last and next_is_software_nv:
                    # Last triplet of round before software_nv round:
                    # Pre-fill nv_A for next round's first triplet (packed into hash free slots).
                    if next_r == 11:  # next round is root
                        nv_fill_ops_for_hash = [
                            ("vbroadcast", self.nv_A + i * VLEN, nv_root_scalar)
                            for i in range(len(first_groups))
                        ]
                    elif next_r == 1 or next_r == rounds - 4:  # next round is level1
                        nv_fill_ops_for_hash = [
                            ("multiply_add", self.nv_A + i * VLEN,
                             idx_base + g * VLEN, v_nv_delta, v_nv_base)
                            for i, g in enumerate(first_groups)
                        ]
                    elif next_r == 2 or next_r == rounds - 3:  # next round is level2
                        # mask_A for first_groups was pre-computed in nav-A of ti=n_triplets-2
                        n_fg = len(first_groups)
                        nv_fill_ops_for_hash = [
                            ("&", mask_B_bufs[j], idx_base + g * VLEN, v_one)
                            for j, g in enumerate(first_groups)
                        ]
                        flow_schedule_for_hash = [None] * 9
                        for j in range(n_fg):
                            flow_schedule_for_hash[j * 2]     = ("vselect", nv_lo_bufs[j], mask_A_bufs[j], v_nv3, v_nv5)
                            flow_schedule_for_hash[j * 2 + 1] = ("vselect", nv_hi_bufs[j], mask_A_bufs[j], v_nv4, v_nv6)
                        for j in range(n_fg):
                            flow_schedule_for_hash[6 + j] = ("vselect", self.nv_A + j * VLEN, mask_B_bufs[j], nv_lo_bufs[j], nv_hi_bufs[j])

                # FR3: XOR bundle handling.
                # ti=0: standalone xor(ti=0) + addr_next(ti=1).
                # ti>=1: xor folded into nav-C of ti-1. No standalone bundle.
                # Spec 04: skip ti=0 xor if pre-applied by previous round's nav.
                if ti == 0:
                    if skip_ti0_xor:
                        # XOR, addr_next, and mask_A were already applied by previous round's
                        # ti=10 nav-A/nav-B (fold optimization). Skip this entire bundle.
                        skip_ti0_xor = False
                    else:
                        xor_ops = [
                            ("^", val_base + g * VLEN, val_base + g * VLEN,
                             nv_cur + i * VLEN)
                            for i, g in enumerate(groups)
                        ]
                        # Spec 04: skip addr_next for software nv rounds (no scatter loads)
                        if software_nv_round:
                            addr_ops = []
                        else:
                            addr_ops = [
                                ("+", self.addr_next + j * VLEN, v_fp,
                                 idx_base + ng * VLEN)
                                for j, ng in enumerate(next_groups)
                            ]
                        # Spec 04 level-2: pre-compute mask_A for next_groups before hash.
                        # xor bundle has 3 free VALU slots (software_nv_round → addr_ops=[]).
                        mask_A_pre = []
                        if is_level2_round and next_groups:
                            mask_A_pre = [
                                ("<", mask_A_bufs[j], idx_base + ng * VLEN, v_five)
                                for j, ng in enumerate(next_groups)
                            ]
                        xor_bundle_valu = xor_ops + addr_ops + mask_A_pre
                        # Optimization: merge xor bundle into preceding flow-only bundle (e.g., pause).
                        # The flow engine is independent of valu; the pause is a no-op with enable_pause=False.
                        # This saves 1 cycle at the setup→main-loop boundary (round 0 ti=0 xor).
                        if (self.instrs and list(self.instrs[-1].keys()) == ["flow"]
                                and len(xor_bundle_valu) <= SLOT_LIMITS["valu"]):
                            self.instrs[-1]["valu"] = xor_bundle_valu
                        else:
                            self.instrs.append({"valu": xor_bundle_valu})

                gi_list  = [idx_base + g * VLEN for g in groups]
                gv_list  = [val_base + g * VLEN for g in groups]
                tn_list  = [tmp_nav_slots[i] for i in range(len(groups))]
                tvld_list = [tmp_vld_slots[i] for i in range(len(groups))]

                # gi-doubler: compute gi = 2*gi + 1 for current triplet's groups.
                # Packed into a free valu cycle within the hash (3rd available slot after nav_d/nav_e).
                # Spec 04 FR4: Skip gi_doubler in the last round (gi_next never used).
                # Spec 04 FR3: Skip gi_doubler in wrap round (gi set directly to 0 in nav-A).
                if is_last_round or is_wrap_round:
                    gi_doubler_ops = None
                else:
                    gi_doubler_ops = [
                        ("multiply_add", gi_list[i], gi_list[i], v_two, v_one)
                        for i in range(len(groups))
                    ]

                # Hash emission: pass deferred nav-D/E from PRIOR triplet to pack in even cycles,
                # and gi_doubler_ops for CURRENT triplet's groups.
                group_addrs = [
                    (val_base + g * VLEN, tmp1_slots[i], tmp2_slots[i])
                    for i, g in enumerate(groups)
                ]
                extra_loads = []
                if is_last:
                    # For non-last rounds: preload nv_A for round r+1's triplet 0.
                    # Spec 04: Skip preload if next round uses software nv (level1/root).
                    preload_n = len(first_groups) if (_r < rounds - 1 and not next_is_software_nv) else 0
                    # 2 loads (vi=6,7 of group 2) were pre-emitted in ti=n_triplets-2's nav-B.
                    _preloaded = 2 if nv_preloaded_early else 0
                    nv_preloaded_early = False
                    if prev_nav_d_bounds or prev_nav_e or preload_n > 0 or nv_fill_ops_for_hash or flow_schedule_for_hash:
                        extra_loads = self.emit_triplet_hash_with_prefetch_and_nav_tail(
                            group_addrs, self.nv_A, self.addr_next, preload_n,
                            prev_nav_d_bounds, prev_nav_e, gi_doubler_ops,
                            nv_fill_ops=nv_fill_ops_for_hash,
                            flow_schedule=flow_schedule_for_hash,
                            preloaded_loads=_preloaded
                        )
                    else:
                        self.emit_triplet_hash(group_addrs, gi_doubler_ops,
                                               nv_fill_ops=nv_fill_ops_for_hash,
                                               flow_schedule=flow_schedule_for_hash)
                else:
                    extra_loads = self.emit_triplet_hash_with_prefetch_and_nav_tail(
                        group_addrs, nv_next_buf, self.addr_next, scatter_n_next,
                        prev_nav_d_bounds, prev_nav_e, gi_doubler_ops,
                        nv_fill_ops=nv_fill_ops_for_hash,
                        flow_schedule=flow_schedule_for_hash
                    )

                # In the last round: enqueue store ops for current triplet's groups.
                # val[g] is finalized after the hash (XOR happened in xor bundle before hash).
                # Stores will be injected into nav bundles below using inject_stores().
                if is_last_round:
                    for g in groups:
                        pending_stores.append(("vstore", addr_arr + g, val_base + g * VLEN))

                # ── Spec 04 FR4: Last round nav skip ──────────────────────────
                # In the last round, gi_next is never used. Replace full nav with
                # minimal cycles: just emit extra loads, xor, and addr_next2.
                # Must respect load ordering: extra loads complete BEFORE xor reads nv_next_buf.
                if is_last_round:
                    has_overflow_lr = len(extra_loads) >= 3 and not is_last
                    n_xor_in_lr = len(next_groups)
                    deferred_xor_lr = None
                    if has_overflow_lr:
                        n_xor_in_lr = len(next_groups) - 1
                        deferred_xor_lr = (len(next_groups) - 1, next_groups[-1])

                    # Cycle 1: extra_loads[0] (if any) + stores
                    # (replaces nav-A; skip gi & v_one computation)
                    if len(extra_loads) > 0 or pending_stores:
                        c1_bundle = {}
                        if len(extra_loads) > 0:
                            c1_bundle["load"] = extra_loads[0]
                        inject_stores(c1_bundle, pending_stores)
                        if c1_bundle:
                            self.instrs.append(c1_bundle)

                    if has_overflow_lr:
                        # Cycle 2: xor for first n_xor_in_lr groups + extra_loads[1] + stores
                        c2_valu = []
                        if n_xor_in_lr > 0:
                            c2_valu = [
                                ("^", val_base + ng * VLEN, val_base + ng * VLEN,
                                 nv_next_buf + j * VLEN)
                                for j, ng in enumerate(next_groups[:n_xor_in_lr])
                            ]
                        c2_bundle = {"valu": c2_valu} if c2_valu else {}
                        if len(extra_loads) > 1:
                            c2_bundle["load"] = extra_loads[1]
                        inject_stores(c2_bundle, pending_stores)
                        if c2_bundle:
                            self.instrs.append(c2_bundle)

                        # Cycle 3: addr_next2 + extra_loads[2] + stores
                        nn_a = (ti + 2) * 3
                        next_next_groups = [g for g in [nn_a, nn_a+1, nn_a+2] if g < n_groups]
                        c3_valu = []
                        if next_next_groups:
                            c3_valu = [
                                ("+", self.addr_next + j * VLEN, v_fp,
                                 idx_base + ng * VLEN)
                                for j, ng in enumerate(next_next_groups)
                            ]
                        if c3_valu or len(extra_loads) > 2 or pending_stores:
                            c3_bundle = {"valu": c3_valu} if c3_valu else {}
                            if len(extra_loads) > 2:
                                c3_bundle["load"] = extra_loads[2]
                            inject_stores(c3_bundle, pending_stores)
                            if c3_bundle:
                                self.instrs.append(c3_bundle)

                        # Cycle 4: deferred xor for last overflow group + stores
                        if deferred_xor_lr is not None:
                            j_def, ng_def = deferred_xor_lr
                            dx_bundle = {"valu": [
                                ("^", val_base + ng_def * VLEN, val_base + ng_def * VLEN,
                                 nv_next_buf + j_def * VLEN)
                            ]}
                            inject_stores(dx_bundle, pending_stores)
                            self.instrs.append(dx_bundle)
                    else:
                        # No overflow: xor + extra_loads[1:] + addr_next2
                        if not is_last and len(next_groups) > 0:
                            xor_next_ops = [
                                ("^", val_base + ng * VLEN, val_base + ng * VLEN,
                                 nv_next_buf + j * VLEN)
                                for j, ng in enumerate(next_groups)
                            ]
                            xor_bundle = {"valu": xor_next_ops}
                            if len(extra_loads) > 1:
                                xor_bundle["load"] = extra_loads[1]
                            inject_stores(xor_bundle, pending_stores)
                            self.instrs.append(xor_bundle)

                        # Remaining extra loads
                        for el in extra_loads[2:]:
                            el_bundle = {"load": el}
                            inject_stores(el_bundle, pending_stores)
                            self.instrs.append(el_bundle)

                        # addr_next2
                        if not is_last:
                            nn_a = (ti + 2) * 3
                            next_next_groups = [g for g in [nn_a, nn_a+1, nn_a+2] if g < n_groups]
                            if next_next_groups:
                                addr_ops = [
                                    ("+", self.addr_next + j * VLEN, v_fp,
                                     idx_base + ng * VLEN)
                                    for j, ng in enumerate(next_next_groups)
                                ]
                                addr_bundle = {"valu": addr_ops}
                                inject_stores(addr_bundle, pending_stores)
                                self.instrs.append(addr_bundle)
                            elif pending_stores:
                                st_bundle = {}
                                inject_stores(st_bundle, pending_stores)
                                if st_bundle:
                                    self.instrs.append(st_bundle)

                    # Drain remaining stores
                    while pending_stores:
                        st_bundle = {}
                        inject_stores(st_bundle, pending_stores)
                        if st_bundle:
                            self.instrs.append(st_bundle)
                        else:
                            break

                    # No nav-D/E deferred (last round)
                    prev_nav_d_bounds = []
                    prev_nav_e = []
                    continue  # Skip gi doubling and bounds check

                # ── Navigation (gi-doubler in hash) ──
                # gi-doubler in hash computes gi = 2*old_gi + 1.
                # For no-overflow cases (n_next≤2): navigation is 2 cycles (nav-A, nav-B).
                # For overflow cases (n_next=3): navigation is 3 cycles (nav-A, nav-B, nav-C) +
                #   addr_next2 (1 cycle), same structure as pre-gi-doubler but nav-B is lighter.
                #
                # Extra loads for n_next=3: split across nav-A (vi=2,3), nav-B (vi=4,5),
                #   nav-C (vi=6,7). Deferred_xor for group j goes into addr_next2 (AFTER nav-C
                #   so vi=6,7 are committed).
                has_overflow = len(extra_loads) >= 3 and not is_last

                # For overflow cases: n_xor_in_nav_c deferred xor
                n_xor_in_nav_c = len(next_groups)
                deferred_xor_group = None
                if has_overflow:
                    n_xor_in_nav_c = len(next_groups) - 1
                    deferred_xor_group = (len(next_groups) - 1, next_groups[-1])

                # nav-A: tmp_nav = gv & v_one (reads hash output, must be after hash)
                # FR3 (wrap round): directly set gi=0 via vbroadcast instead of &v_one + nav-D/E.
                # For ti=n_triplets-2 in non-last rounds: fold prologue addr_next computation.
                if is_wrap_round:
                    # FR3: all inputs wrap to gi=0; set directly, skip bounds check + multiply.
                    nav_a_ops = [("vbroadcast", gi_list[i], c0) for i in range(len(groups))]
                else:
                    nav_a_ops = [("&", tn_list[i], gv_list[i], v_one) for i in range(len(groups))]
                if ti == n_triplets - 2 and _r < rounds - 1:
                    # Spec 04: skip prologue addr_next if next round uses software nv
                    if not next_is_software_nv:
                        prologue_addr_ops = [
                            ("+", self.addr_next + j * VLEN, v_fp, idx_base + ng * VLEN)
                            for j, ng in enumerate(first_groups)
                        ]
                        nav_a_ops = nav_a_ops + prologue_addr_ops
                    elif next_r == 2 or next_r == rounds - 3:
                        # Spec 04 level-2: pre-compute mask_A for first_groups before last hash.
                        # These free slots are available since prologue addr_next is skipped.
                        nav_a_ops = nav_a_ops + [
                            ("<", mask_A_bufs[j], idx_base + g * VLEN, v_five)
                            for j, g in enumerate(first_groups)
                        ]
                # Spec 04 level-2: pre-compute mask_A for ti+2's groups (= ti+1's next_groups)
                # in ti's nav-A, so they're ready before ti+1's hash starts.
                # (ti=0's xor bundle already handles mask_A for ti=0's next_groups.)
                if is_level2_round and not is_last:
                    nn_a_future = (ti + 2) * 3
                    future_groups_l2 = [g for g in [nn_a_future, nn_a_future + 1, nn_a_future + 2] if g < n_groups]
                    if future_groups_l2:
                        nav_a_ops = nav_a_ops + [
                            ("<", mask_A_bufs[j], idx_base + ng * VLEN, v_five)
                            for j, ng in enumerate(future_groups_l2)
                        ]
                # Fold next round's ti=0 xor bundle: pre-compute addr_next or mask_A in nav-A.
                # Applies only to ti=10 (is_last) of non-last rounds.
                # This lets us skip the entire ti=0 xor bundle in the next round.
                # IMPORTANT: Two hazards for addr_next slot j=2 (addr_next+16..23):
                # 1. addr_next+16..23 must NOT be written in nav-A because extra_loads[1,2]
                #    read addr_next+20..23 in nav-B and after; they must see OLD values.
                # 2. The XOR for group 2 (val ^= nv_A[16..23]) must NOT happen until
                #    nv_A[20..23] (lanes 4-7) are fully loaded. The loads happen in:
                #    - nav_b (extra_loads[1] = vi=4,5 → nv_A[20,21])
                #    - standalone (extra_loads[2] = vi=6,7 → nv_A[22,23])
                #    So the XOR for group 2 must be deferred to AFTER extra_loads[2].
                #    Groups 0 and 1 XOR can happen in nav-B (their nv fully loaded by then).
                # Strategy:
                # - nav-A: addr_next for j=0,1 only (groups 3,4)
                # - nav-B: XOR for groups 0,1 only; extra_loads[1] in load slot
                # - standalone extra_loads[2]: load nv_A[22,23]
                # - deferred bundle: addr_next+16..23 for group 5 AND XOR for group 2
                deferred_addr_next_j2 = None  # ("+", addr_next+2*VLEN, v_fp, idx_base+5*VLEN) if needed
                deferred_xor_g2 = None        # ("^", val_base+2*VLEN, val_base+2*VLEN, nv_A+2*VLEN) if needed
                if is_last and _r < rounds - 1:
                    next_r_is_level2 = (next_r == 2 or next_r == rounds - 3)
                    if not next_is_software_nv:
                        # Pre-compute addr_next for next round's ti=1 groups [3,4] in nav-A (safe).
                        # Defer group 5 (j=2 → addr_next+16..23) to after all extra_loads complete,
                        # to avoid overwriting addr_next+16..23 before extra_loads[1,2] use it.
                        nav_a_ops = nav_a_ops + [
                            ("+", self.addr_next + j * VLEN, v_fp, idx_base + ng * VLEN)
                            for j, ng in enumerate([3, 4])   # only j=0,1
                        ]
                        deferred_addr_next_j2 = ("+", self.addr_next + 2 * VLEN, v_fp, idx_base + 5 * VLEN)
                        # Defer XOR for group 2 (nv_A[16..23] lanes 4-7 not loaded until after nav-B)
                        if len(first_groups) > 2:
                            deferred_xor_g2 = ("^", val_base + first_groups[2] * VLEN, val_base + first_groups[2] * VLEN, self.nv_A + 2 * VLEN)
                    elif next_r_is_level2:
                        # Pre-compute mask_A for next round's ti=0's next_groups [3,4,5]
                        nav_a_ops = nav_a_ops + [
                            ("<", mask_A_bufs[j], idx_base + ng * VLEN, v_five)
                            for j, ng in enumerate([3, 4, 5])
                        ]
                nav_a_bundle = {"valu": nav_a_ops}
                if len(extra_loads) > 0:
                    nav_a_bundle["load"] = extra_loads[0]
                # For wrap round non-overflow: defer nav-A emission, will merge into nav-B.
                if not (is_wrap_round and not has_overflow):
                    inject_stores(nav_a_bundle, pending_stores)
                    self.instrs.append(nav_a_bundle)

                if has_overflow:
                    # 3-cycle nav (old nav-B style but gi already doubled):
                    # nav-B: gi += tmp_nav + xor for first n_xor_in_nav_c groups
                    # FR3: skip gi update (gi already set to 0 in nav-A via vbroadcast)
                    nav_b_ops = [] if is_wrap_round else [("+", gi_list[i], gi_list[i], tn_list[i]) for i in range(len(groups))]
                    if not is_last and n_xor_in_nav_c > 0:
                        xor_next_ops = [
                            ("^", val_base + ng * VLEN, val_base + ng * VLEN,
                             nv_next_buf + j * VLEN)
                            for j, ng in enumerate(next_groups[:n_xor_in_nav_c])
                        ]
                        nav_b_ops = nav_b_ops + xor_next_ops
                    nav_b_bundle = {"valu": nav_b_ops}
                    if len(extra_loads) > 1:
                        nav_b_bundle["load"] = extra_loads[1]
                    inject_stores(nav_b_bundle, pending_stores)
                    self.instrs.append(nav_b_bundle)

                    # nav-C: addr_next2_ops in valu + extra_loads[2] in load slot
                    nn_a = (ti + 2) * 3
                    next_next_groups = [g for g in [nn_a, nn_a+1, nn_a+2] if g < n_groups]
                    nav_c_valu_ops = []
                    # Spec 04: skip addr_next2 for software nv rounds
                    if next_next_groups and not software_nv_round:
                        nav_c_valu_ops = [
                            ("+", self.addr_next + j * VLEN, v_fp,
                             idx_base + ng * VLEN)
                            for j, ng in enumerate(next_next_groups)
                        ]
                    if nav_c_valu_ops or len(extra_loads) > 2 or pending_stores:
                        nav_c_bundle = {"valu": nav_c_valu_ops} if nav_c_valu_ops else {"valu": []}
                        if len(extra_loads) > 2:
                            nav_c_bundle["load"] = extra_loads[2]
                        inject_stores(nav_c_bundle, pending_stores)
                        if nav_c_bundle.get("valu") or nav_c_bundle.get("load") or nav_c_bundle.get("store"):
                            self.instrs.append(nav_c_bundle)

                    # addr_next2 = deferred_xor (now standalone, after vi=6,7 committed in nav-C)
                    if deferred_xor_group is not None:
                        j_def, ng_def = deferred_xor_group
                        dx_bundle = {"valu": [
                            ("^", val_base + ng_def * VLEN, val_base + ng_def * VLEN,
                             nv_next_buf + j_def * VLEN)
                        ]}
                        inject_stores(dx_bundle, pending_stores)
                        self.instrs.append(dx_bundle)
                else:
                    # No overflow: nav-B (nav-A already emitted above, unless wrap merge)
                    # FR3: skip gi update for wrap round (gi already set to 0 in nav-A)
                    nav_b_ops = [] if is_wrap_round else [("+", gi_list[i], gi_list[i], tn_list[i]) for i in range(len(groups))]
                    if not is_last and len(next_groups) > 0:
                        xor_next_ops = [
                            ("^", val_base + ng * VLEN, val_base + ng * VLEN,
                             nv_next_buf + j * VLEN)
                            for j, ng in enumerate(next_groups)
                        ]
                        nav_b_ops = nav_b_ops + xor_next_ops
                    # Fold next round's ti=0 xor into nav-B (only for ti=10 = is_last)
                    # For group 2 (i=2): nv_A[16..23] lanes 4-7 are loaded in nav-B's load slot
                    # (extra_loads[1]) and after, so the XOR for group 2 must be deferred.
                    # Groups 0 and 1 are safe: their nv fully loaded before nav-B.
                    if is_last and _r < rounds - 1:
                        # Determine which groups can be XORed now (nv fully loaded)
                        safe_xor_groups = [(i, g) for i, g in enumerate(first_groups) if deferred_xor_g2 is None or i < 2]
                        nav_b_ops = nav_b_ops + [
                            ("^", val_base + g * VLEN, val_base + g * VLEN,
                             self.nv_A + i * VLEN)
                            for i, g in safe_xor_groups
                        ]
                    # For wrap round (non-overflow): merge deferred nav-A ops into nav-B if it fits.
                    # If merging would exceed SLOT_LIMITS["valu"], emit nav-A separately first.
                    extra_load_start = 0 if is_wrap_round else 1
                    if is_wrap_round:
                        if len(nav_a_ops) + len(nav_b_ops) <= SLOT_LIMITS["valu"]:
                            nav_b_ops = nav_a_ops + nav_b_ops
                        else:
                            # Too many ops to merge: emit nav-A separately.
                            na_bundle = {"valu": nav_a_ops}
                            if len(extra_loads) > 0:
                                na_bundle["load"] = extra_loads[0]
                            inject_stores(na_bundle, pending_stores)
                            self.instrs.append(na_bundle)
                            # nav-B will use extra_loads[1] now that extra_loads[0] is consumed.
                            extra_load_start = 1
                    nav_b_bundle = {"valu": nav_b_ops}
                    if len(extra_loads) > extra_load_start:
                        nav_b_bundle["load"] = extra_loads[extra_load_start]
                    # At ti=n_triplets-2: pre-load vi=6,7 of nv_A group 2 into nav-B's free
                    # load slot. addr_next+2*VLEN was just computed in nav-A (prologue_addr_ops).
                    # This eliminates the standalone load cycle at ti=n_triplets-1.
                    elif (ti == n_triplets - 2 and _r < rounds - 1 and not next_is_software_nv):
                        nav_b_bundle["load"] = [
                            ("load_offset", self.nv_A + 2 * VLEN, self.addr_next + 2 * VLEN, 6),
                            ("load_offset", self.nv_A + 2 * VLEN, self.addr_next + 2 * VLEN, 7),
                        ]
                        nv_preloaded_early = True
                    inject_stores(nav_b_bundle, pending_stores)
                    self.instrs.append(nav_b_bundle)

                    # Emit any remaining extra_loads (e.g., from last-triplet preload overflow)
                    for el in extra_loads[2:]:
                        el_bundle = {"load": el}
                        inject_stores(el_bundle, pending_stores)
                        self.instrs.append(el_bundle)

                    # Emit deferred bundle AFTER all extra_loads complete:
                    # - addr_next+16..23 for group 5 (safe since extra_loads[1,2] done)
                    # - XOR for group 2 (all of nv_A[16..23] now loaded)
                    deferred_ops = []
                    if deferred_addr_next_j2 is not None:
                        deferred_ops.append(deferred_addr_next_j2)
                    if deferred_xor_g2 is not None:
                        deferred_ops.append(deferred_xor_g2)
                    if deferred_ops:
                        deferred_bundle = {"valu": deferred_ops}
                        inject_stores(deferred_bundle, pending_stores)
                        self.instrs.append(deferred_bundle)

                    # For no-overflow non-last triplets: still need addr_next2 if next_next exists
                    if not is_last:
                        nn_a = (ti + 2) * 3
                        next_next_groups = [g for g in [nn_a, nn_a+1, nn_a+2] if g < n_groups]
                        # Spec 04: skip addr_next2 for software nv rounds
                        if next_next_groups and not software_nv_round:
                            addr_ops = [
                                ("+", self.addr_next + j * VLEN, v_fp,
                                 idx_base + ng * VLEN)
                                for j, ng in enumerate(next_next_groups)
                            ]
                            addr_bundle = {"valu": addr_ops}
                            inject_stores(addr_bundle, pending_stores)
                            self.instrs.append(addr_bundle)
                        elif pending_stores:
                            # No addr_next2 needed, but still have pending stores to inject
                            st_bundle = {}
                            inject_stores(st_bundle, pending_stores)
                            if st_bundle:
                                self.instrs.append(st_bundle)

                # nav-D bounds: deferred to hi=0 even of ti+1's hash
                # FR3: gi=0 is already correct (all wrap); skip bounds check and multiply.
                if is_wrap_round:
                    curr_nav_d_bounds = []
                    curr_nav_e = []
                else:
                    curr_nav_d_bounds = [
                        ("<", tvld_list[i], gi_list[i], v_nnodes) for i in range(len(groups))
                    ]
                    # nav-E: deferred to hi=1 even of ti+1's hash
                    curr_nav_e = [
                        ("*", gi_list[i], gi_list[i], tvld_list[i]) for i in range(len(groups))
                    ]

                if is_last and _r == rounds - 1:
                    # Last triplet of last round: idx is not checked by tests,
                    # so skip nav-D/E (saves 2 cycles per kernel run).
                    prev_nav_d_bounds = []
                    prev_nav_e = []
                elif is_last:
                    # Last triplet but not last round: defer nav-D/E into next round's ti=0 hash.
                    # Also signal that next round's ti=0 xor bundle was pre-applied (fold opt).
                    prev_nav_d_bounds = curr_nav_d_bounds
                    prev_nav_e = curr_nav_e
                    skip_ti0_xor = True
                else:
                    prev_nav_d_bounds = curr_nav_d_bounds
                    prev_nav_e = curr_nav_e

        # ── Store final val back to memory ─────────────────────────────────────
        # Stores are injected into the last round's nav bundles above (inject_stores).
        # Any remaining pending stores (if nav cycles weren't enough) go here.
        # For the standard case (n_groups=32, 11 triplets, ~44 nav cycles), all 32 stores
        # fit within nav cycles, so this loop emits nothing.
        # addr_arr[g] = inp_values_p + g*VLEN (set during Phase 2a, never modified after).
        while pending_stores:
            take = min(STORE_LIMIT, len(pending_stores))
            store_ops = [pending_stores.pop(0) for _ in range(take)]
            self.instrs.append({"store": store_ops})

        # No final pause: machine stops when pc >= len(program). pause costs 1 cycle.

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
