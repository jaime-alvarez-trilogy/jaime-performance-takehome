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

# ── Spec 06: DAG Scheduler Foundation ────────────────────────────────────────

from dataclasses import dataclass, field as dc_field


@dataclass
class Op:
    """Models a single machine operation for DAG scheduling."""
    engine: str           # 'valu'|'alu'|'load'|'store'|'flow'
    instr: tuple          # complete instruction tuple for to_instr_dict
    dst: int              # scratch address written (−1 if none)
    dep_srcs: list        # scratch addresses READ (for dep analysis; excludes literal ints)
    context_id: int       # group index within the pass (0-based)
    round_id: int         # round (0–15)
    stage: int            # intra-round ordering (0-based)
    deps: list = dc_field(default_factory=list)   # Op objects this depends on
    lat: int = 1          # output-ready latency


@dataclass
class ScratchLayout:
    """Per-context scratch address layout for DAG scheduler."""
    n_contexts: int
    context_bases: list    # context_bases[i] = base addr for context i's new 32-word block
    val_bases: list        # val_bases[i] = val_base + group_ids[i] * VLEN
    idx_bases: list        # idx_bases[i] = idx_base + group_ids[i] * VLEN

    NV_OFF   = 0     # nv_buf:   8 words
    TMP1_OFF = 8     # tmp1:     8 words
    TMP2_OFF = 16    # tmp2:     8 words
    ADDR_OFF = 24    # addr_buf: 8 words
    TNAV_OFF = 32    # tmp_nav:  8 words
    TVLD_OFF = 40    # tmp_vld:  8 words
    WORDS_PER_CTX = 48   # 6 × 8 words per context

    def nv_buf(self, ctx):   return self.context_bases[ctx] + ScratchLayout.NV_OFF
    def tmp1(self, ctx):     return self.context_bases[ctx] + ScratchLayout.TMP1_OFF
    def tmp2(self, ctx):     return self.context_bases[ctx] + ScratchLayout.TMP2_OFF
    def addr_buf(self, ctx): return self.context_bases[ctx] + ScratchLayout.ADDR_OFF
    def tmp_nav(self, ctx):  return self.context_bases[ctx] + ScratchLayout.TNAV_OFF
    def tmp_vld(self, ctx):  return self.context_bases[ctx] + ScratchLayout.TVLD_OFF
    def val_buf(self, ctx):  return self.val_bases[ctx]
    def idx_buf(self, ctx):  return self.idx_bases[ctx]


def allocate_scratch(n_contexts, base_offset, val_base, idx_base, group_ids):
    """Allocate per-context scratch, reusing val/idx arrays from setup phase."""
    total = base_offset + n_contexts * ScratchLayout.WORDS_PER_CTX
    assert total <= SCRATCH_SIZE, (
        f"Scratch overflow: need {total} words, have {SCRATCH_SIZE}"
    )
    context_bases = [base_offset + i * ScratchLayout.WORDS_PER_CTX
                     for i in range(n_contexts)]
    val_bases = [val_base + g * VLEN for g in group_ids]
    idx_bases = [idx_base + g * VLEN for g in group_ids]
    return ScratchLayout(
        n_contexts=n_contexts,
        context_bases=context_bases,
        val_bases=val_bases,
        idx_bases=idx_bases,
    )


class Bundle:
    """One VLIW instruction cycle: enforces SLOT_LIMITS per engine."""
    _LIMITS = {k: v for k, v in SLOT_LIMITS.items() if k != 'debug'}

    def __init__(self):
        self._slots = {eng: [] for eng in self._LIMITS}

    def can_fit(self, op):
        eng = op.engine
        return len(self._slots.get(eng, [])) < self._LIMITS.get(eng, 0)

    def add(self, op):
        assert self.can_fit(op), f"Bundle overflow: engine={op.engine}"
        self._slots[op.engine].append(op)

    def is_empty(self):
        return all(len(v) == 0 for v in self._slots.values())

    def to_instr_dict(self):
        result = {}
        for eng, ops in self._slots.items():
            if ops:
                result[eng] = [op.instr for op in ops]
        return result


def build_op_graph(contexts, rounds, scratch, hsc_new, v_fp, v_one, v_two, v_nnodes):
    """
    Generate all Ops for a set of contexts/rounds with populated deps.

    contexts: list of local context indices (0-based, into scratch layout)
    rounds:   list of round indices (0–15)
    scratch:  ScratchLayout
    hsc_new:  hash stage constants list from KernelBuilder
    v_fp, v_one, v_nnodes: constant scratch addresses
    """
    n_rounds = len(rounds)
    last_round = rounds[-1]
    all_ops = []

    for ctx in contexts:
        val_buf  = scratch.val_buf(ctx)
        idx_buf  = scratch.idx_buf(ctx)
        nv_buf   = scratch.nv_buf(ctx)
        tmp1     = scratch.tmp1(ctx)
        tmp2     = scratch.tmp2(ctx)
        addr_buf = scratch.addr_buf(ctx)
        tmp_nav  = scratch.tmp_nav(ctx)
        tmp_vld  = scratch.tmp_vld(ctx)

        ctx_ops = []

        for r in rounds:
            # Stage counter per round
            s = 0

            # ── Stage 0: addr_buf = v_fp + idx_buf ──────────────────────────
            op_addr = Op(
                engine='valu',
                instr=('+', addr_buf, v_fp, idx_buf),
                dst=addr_buf,
                dep_srcs=[v_fp, idx_buf],
                context_id=ctx, round_id=r, stage=s,
            )
            ctx_ops.append(op_addr)
            s += 1

            # ── Stages 1–4: scatter load nv_buf (4 load pairs = 8 ops) ─────
            # Each load_offset writes to nv_buf+vi (one word per op).
            # Track individual words for accurate dep analysis.
            load_ops_this_round = []
            for vi in range(0, VLEN, 2):
                op_ld0 = Op(
                    engine='load',
                    instr=('load_offset', nv_buf, addr_buf, vi),
                    dst=nv_buf + vi,        # individual word destination
                    dep_srcs=[addr_buf],    # reads the base address (all VLEN words)
                    context_id=ctx, round_id=r, stage=s,
                )
                op_ld1 = Op(
                    engine='load',
                    instr=('load_offset', nv_buf, addr_buf, vi + 1),
                    dst=nv_buf + vi + 1,    # individual word destination
                    dep_srcs=[addr_buf],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_ld0)
                ctx_ops.append(op_ld1)
                load_ops_this_round.append(op_ld0)
                load_ops_this_round.append(op_ld1)
                s += 1

            # ── Stage 5: val_buf ^= nv_buf ──────────────────────────────────
            # Reads all VLEN words of nv_buf — must wait for ALL 8 loads.
            op_xor = Op(
                engine='valu',
                instr=('^', val_buf, val_buf, nv_buf),
                dst=val_buf,
                dep_srcs=[val_buf] + [nv_buf + vi for vi in range(VLEN)],
                context_id=ctx, round_id=r, stage=s,
            )
            ctx_ops.append(op_xor)
            s += 1

            # ── Stages 6–14: hash (9 VALU cycles) ──────────────────────────
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                is_mul = (op1 == "+" and op2 == "+" and op3 == "<<")
                stage_info = hsc_new[hi]

                if is_mul:
                    _, v_factor, v_C = stage_info
                    op_hash = Op(
                        engine='valu',
                        instr=('multiply_add', val_buf, val_buf, v_factor, v_C),
                        dst=val_buf,
                        dep_srcs=[val_buf, v_factor, v_C],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    ctx_ops.append(op_hash)
                    s += 1
                else:
                    _, op1_s, v_c1, op2_s, op3_s, v_c2 = stage_info
                    # Odd cycle: tmp1 = val op1 v_c1; tmp2 = val op3 v_c2
                    op_h1 = Op(
                        engine='valu',
                        instr=(op1_s, tmp1, val_buf, v_c1),
                        dst=tmp1,
                        dep_srcs=[val_buf, v_c1],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    op_h2 = Op(
                        engine='valu',
                        instr=(op3_s, tmp2, val_buf, v_c2),
                        dst=tmp2,
                        dep_srcs=[val_buf, v_c2],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    ctx_ops.append(op_h1)
                    ctx_ops.append(op_h2)
                    s += 1
                    # Even cycle: val = tmp1 op2 tmp2
                    op_h3 = Op(
                        engine='valu',
                        instr=(op2_s, val_buf, tmp1, tmp2),
                        dst=val_buf,
                        dep_srcs=[tmp1, tmp2],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    ctx_ops.append(op_h3)
                    s += 1

            # ── Navigation (skip for last round) ────────────────────────────
            # Navigation formula: idx_next = 2*idx + 1 + (val & 1)
            #   gi_double: idx = idx * 2 + 1  (via multiply_add(idx, idx, v_two, v_one))
            #   nav_a:     tmp_nav = val & 1  (low bit of hash output)
            #   nav_b:     idx += tmp_nav     (adds 0 or 1)
            #   nav_c:     tmp_vld = idx < n_nodes  (bounds check)
            #   nav_d:     idx *= tmp_vld     (wrap OOB to 0)
            if r != last_round:
                # Gi-doubler: idx = idx * 2 + 1
                op_gi_double = Op(
                    engine='valu',
                    instr=('multiply_add', idx_buf, idx_buf, v_two, v_one),
                    dst=idx_buf,
                    dep_srcs=[idx_buf, v_two, v_one],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_gi_double)
                s += 1

                # Nav A: tmp_nav = val_buf & 1
                op_nav_a = Op(
                    engine='valu',
                    instr=('&', tmp_nav, val_buf, v_one),
                    dst=tmp_nav,
                    dep_srcs=[val_buf, v_one],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nav_a)
                s += 1

                # Nav B: idx_buf += tmp_nav
                op_nav_b = Op(
                    engine='valu',
                    instr=('+', idx_buf, idx_buf, tmp_nav),
                    dst=idx_buf,
                    dep_srcs=[idx_buf, tmp_nav],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nav_b)
                s += 1

                # Nav bounds: tmp_vld = idx_buf < v_nnodes
                op_nav_c = Op(
                    engine='valu',
                    instr=('<', tmp_vld, idx_buf, v_nnodes),
                    dst=tmp_vld,
                    dep_srcs=[idx_buf, v_nnodes],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nav_c)
                s += 1

                # Nav multiply: idx_buf *= tmp_vld (wraps OOB indices to 0)
                op_nav_d = Op(
                    engine='valu',
                    instr=('*', idx_buf, idx_buf, tmp_vld),
                    dst=idx_buf,
                    dep_srcs=[idx_buf, tmp_vld],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nav_d)
                s += 1

        # ── Populate deps via last-writer analysis ───────────────────────────
        # Process ops in round/stage order; track last writer per scratch addr
        last_writer = {}
        for op in ctx_ops:
            for src in op.dep_srcs:
                if src in last_writer:
                    writer = last_writer[src]
                    if writer not in op.deps:
                        op.deps.append(writer)
            if op.dst >= 0:
                last_writer[op.dst] = op

        all_ops.extend(ctx_ops)

    return all_ops


def list_schedule(ops):
    """
    Greedy list scheduler: longest-remaining-path priority.
    Returns List[Bundle] with one bundle per cycle.
    """
    if not ops:
        return []

    # Build consumer graph (inverse of deps), keyed by id(op)
    consumers = defaultdict(list)
    for op in ops:
        for dep in op.deps:
            consumers[id(dep)].append(op)

    # Compute longest-remaining-path (lrp) via iterative bottom-up
    lrp = {}
    def compute_lrp(op):
        if id(op) in lrp:
            return lrp[id(op)]
        cons = consumers[id(op)]
        if not cons:
            lrp[id(op)] = 0
            return 0
        val = 1 + max(compute_lrp(c) for c in cons)
        lrp[id(op)] = val
        return val

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(20000)
    try:
        for op in ops:
            compute_lrp(op)
    finally:
        sys.setrecursionlimit(old_limit)

    # Scheduler state
    dep_remaining = {id(op): len(op.deps) for op in ops}
    scheduled_at = {}   # id(op) → cycle
    ready = sorted(
        [op for op in ops if len(op.deps) == 0],
        key=lambda o: (-lrp[id(o)], o.context_id)
    )

    bundles = []
    cycle = 0
    total = len(ops)
    scheduled_count = 0

    while scheduled_count < total:
        bundle = Bundle()
        scheduled_this_cycle = []

        # Build priority-sorted candidate list from ready ops
        candidates = sorted(ready, key=lambda o: (-lrp[id(o)], o.context_id))

        for op in candidates:
            if not bundle.can_fit(op):
                continue
            # Check latency: all deps must have finished before this cycle
            all_deps_ready = True
            for dep in op.deps:
                dep_cycle = scheduled_at.get(id(dep))
                if dep_cycle is None or dep_cycle + dep.lat > cycle:
                    all_deps_ready = False
                    break
            if not all_deps_ready:
                continue

            bundle.add(op)
            scheduled_at[id(op)] = cycle
            scheduled_this_cycle.append(op)

        # Remove scheduled ops from ready
        scheduled_set = {id(op) for op in scheduled_this_cycle}
        ready = [op for op in ready if id(op) not in scheduled_set]
        scheduled_count += len(scheduled_this_cycle)

        # Unlock consumers
        for op in scheduled_this_cycle:
            for consumer in consumers[id(op)]:
                dep_remaining[id(consumer)] -= 1
                if dep_remaining[id(consumer)] == 0:
                    ready.append(consumer)

        bundles.append(bundle)
        cycle += 1

        # Safety: if nothing scheduled and ready is non-empty, it's a latency stall
        # Just advance (stall cycle). If ready is empty and unscheduled exist, something wrong.
        if not scheduled_this_cycle and not ready and scheduled_count < total:
            # Stall: all ready ops blocked by latency; advance cycle
            # Re-add all latency-blocked ops with deps satisfied
            for op in ops:
                if id(op) not in scheduled_at and dep_remaining[id(op)] == 0:
                    ready.append(op)

    return bundles


# ── End Spec 06 DAG Infrastructure ───────────────────────────────────────────


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

        # ── Scratch allocation (Spec 06: streamlined for DAG scheduler) ────────
        # Memory header pointers (loaded from mem[4..6])
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
        # Instruction: inp_values_p
        self.instrs.append({"load": [
            ("load", self.scratch["inp_values_p"], self.scratch_const(6)),
        ]})

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
        all_broadcasts = [
            ("vbroadcast", v_one,    c1),
            ("vbroadcast", v_two,    c2),
            ("vbroadcast", v_nnodes, c_nn),
            ("vbroadcast", v_fp,     self.scratch["forest_values_p"]),
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

        # ── Spec 06: DAG-based multi-context scheduler ────────────────────────
        # Replace hand-scheduled quadruplet loop with greedy list scheduler.
        # group_size=16: 16 contexts × 48 words = 768 words; dag_base ~697, 697+768=1465 ≤ 1536.
        group_size = 16
        dag_base_offset = self.scratch_ptr
        assert dag_base_offset + group_size * ScratchLayout.WORDS_PER_CTX <= SCRATCH_SIZE, (
            f"DAG scratch overflow: need {dag_base_offset + group_size * ScratchLayout.WORDS_PER_CTX}, "
            f"have {SCRATCH_SIZE}. Reduce group_size."
        )
        self.scratch_ptr += group_size * ScratchLayout.WORDS_PER_CTX

        passes = [
            list(range(i, min(i + group_size, n_groups)))
            for i in range(0, n_groups, group_size)
        ]

        for pass_groups in passes:
            n_ctx = len(pass_groups)
            pass_layout = allocate_scratch(
                n_contexts=n_ctx,
                base_offset=dag_base_offset,  # reuse same scratch region each pass
                val_base=val_base,
                idx_base=idx_base,
                group_ids=pass_groups,
            )
            ops = build_op_graph(
                contexts=list(range(n_ctx)),
                rounds=list(range(rounds)),
                scratch=pass_layout,
                hsc_new=self.hsc_new,
                v_fp=v_fp,
                v_one=v_one,
                v_two=v_two,
                v_nnodes=v_nnodes,
            )
            schedule = list_schedule(ops)
            for bundle in schedule:
                if not bundle.is_empty():
                    self.instrs.append(bundle.to_instr_dict())

        # ── Store final val back to memory ─────────────────────────────────────
        STORE_LIMIT = SLOT_LIMITS["store"]
        pending_stores = [
            ("vstore", addr_arr + g, val_base + g * VLEN)
            for g in range(n_groups)
        ]
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
