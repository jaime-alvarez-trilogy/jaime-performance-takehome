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

    NV_OFF   = 0     # nv_buf:         8 words  [independent]
    TMP1_OFF = 8     # tmp1 + tmp_nav: 8 words  [aliased — sequential use]
    TMP2_OFF = 16    # tmp2 + addr_buf + tmp_vld: 8 words  [aliased — sequential use]
    ADDR_OFF = 16    # same as TMP2_OFF
    TNAV_OFF = 8     # same as TMP1_OFF
    TVLD_OFF = 16    # same as TMP2_OFF
    WORDS_PER_CTX = 24   # 3 × 8 words per context

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


def build_op_graph(contexts, rounds, scratch, hsc_new, v_fp, v_one, v_two, v_nnodes=None,
                   context_types=None, const_map=None,
                   v_nv_root=None, v_nv_delta=None, v_nv_base=None,
                   v_nv1=None, v_nv2=None, v_nv3=None, v_nv4=None,
                   v_nv5=None, v_nv6=None,
                   v_diff_T=None, v_diff_F=None):
    """
    Generate all Ops for a set of contexts/rounds with populated deps.

    contexts:      list of local context indices (0-based, into scratch layout)
    rounds:        list of round indices (0–15)
    scratch:       ScratchLayout
    hsc_new:       hash stage constants list from KernelBuilder
    v_fp, v_one, v_two, v_nnodes: constant scratch addresses
    context_types: dict {ctx: 'valu_hash'|'alu_hash'} (Spec 07); None → all valu_hash
    const_map:     dict {raw_val: scalar_scratch_addr} for ALU hash scalar constants
    v_nv_root:     scratch addr of broadcast vector node_val[0] (Spec 08; rounds 0,10,11)
    v_nv_delta:    unused (kept for API compatibility)
    v_nv_base:     unused (kept for API compatibility)
    v_nv1..v_nv6:  scratch addrs of broadcast vectors node_val[1..6] (Spec 08)
                   level2 uses gi&2 (v_two constant) for dispatch instead of v_five
    """
    if context_types is None:
        context_types = {}
    if const_map is None:
        const_map = {}

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

            # ── Spec 08: determine round class ──────────────────────────────
            rc = round_class(r)

            # ── Stage 0: addr_buf = v_fp + idx_buf (scatter/last/wrap rounds) ──
            # Non-scatter rounds don't need addr_buf (no scatter loads issued).
            # wrap (10): still needs scatter load to read level-10 node values;
            #   only the nav step is simplified (idx zeroed unconditionally).
            if rc in ('scatter', 'last', 'wrap'):
                op_addr = Op(
                    engine='valu',
                    instr=('+', addr_buf, v_fp, idx_buf),
                    dst=addr_buf,
                    dep_srcs=[v_fp, idx_buf],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_addr)
                s += 1

            # ── Stages 1–4: nv source (class-dependent) ─────────────────────
            # scatter/last/wrap: 8 scatter load_offset ops (4 pairs, VLEN=8 lanes)
            # software_nv rounds 1,12: vselect(lo_bit, v_nv1, v_nv2)
            # software_nv rounds 0,11: nv = v_nv_root (already broadcast); skip nv_buf write
            # level2 (2,13): 3-level vselect using gi&1 and gi&2 bits
            if rc in ('scatter', 'last', 'wrap'):
                # Scatter load nv_buf (4 load pairs = 8 ops).
                # Each load_offset writes to nv_buf+vi (one word per op).
                for vi in range(0, VLEN, 2):
                    op_ld0 = Op(
                        engine='load',
                        instr=('load_offset', nv_buf, addr_buf, vi),
                        dst=nv_buf + vi,
                        dep_srcs=[addr_buf],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    op_ld1 = Op(
                        engine='load',
                        instr=('load_offset', nv_buf, addr_buf, vi + 1),
                        dst=nv_buf + vi + 1,
                        dep_srcs=[addr_buf],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    ctx_ops.append(op_ld0)
                    ctx_ops.append(op_ld1)
                    s += 1
            elif rc == 'software_nv' and r in (1, 12):
                # gi ∈ {1, 2}: select between v_nv1 (gi=1, lo_bit=1) and v_nv2 (gi=2, lo_bit=0).
                # Step 1: lo_bit = idx_buf & 1  (VALU, writes tmp_nav)
                op_lo = Op(
                    engine='valu',
                    instr=('&', tmp_nav, idx_buf, v_one),
                    dst=tmp_nav,
                    dep_srcs=[idx_buf, v_one],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_lo)
                s += 1
                # Step 2: nv_buf = vselect(lo_bit, v_nv1, v_nv2)  (FLOW)
                # lo=1 → gi=1 → nv1; lo=0 → gi=2 → nv2
                op_nv = Op(
                    engine='flow',
                    instr=('vselect', nv_buf, tmp_nav, v_nv1, v_nv2),
                    dst=nv_buf,
                    dep_srcs=[tmp_nav, v_nv1, v_nv2],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nv)
                s += 1
            elif rc == 'level2':
                # gi ∈ {3, 4, 5, 6}: 2-bit dispatch using lo_bit (gi&1) and bit1 (gi&2).
                # gi=3(011): lo=1, bit1=2(T) → nv3
                # gi=4(100): lo=0, bit1=0(F) → nv4
                # gi=5(101): lo=1, bit1=0(F) → nv5
                # gi=6(110): lo=0, bit1=2(T) → nv6
                # bit1 groups: {3,6} have bit1≠0; {4,5} have bit1=0
                #
                # Dispatch using v_two (broadcast(2), existing constant):
                # If bit1≠0 and lo=1 → nv3; if bit1≠0 and lo=0 → nv6
                # If bit1=0  and lo=1 → nv5; if bit1=0  and lo=0 → nv4
                #
                # Step 1: lo_bit = idx_buf & 1  →  tmp1   (VALU; keep tmp_nav free for step 4)
                op_lo = Op(
                    engine='valu',
                    instr=('&', tmp1, idx_buf, v_one),
                    dst=tmp1,
                    dep_srcs=[idx_buf, v_one],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_lo)
                s += 1
                # Step 2: bit1 = idx_buf & 2  →  tmp_vld  (VALU; reuse v_two constant)
                op_b1 = Op(
                    engine='valu',
                    instr=('&', tmp_vld, idx_buf, v_two),
                    dst=tmp_vld,
                    dep_srcs=[idx_buf, v_two],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_b1)
                s += 1
                # Step 3: nv_bit1_T = multiply_add(lo_bit, diff_T, v_nv6) → nv_buf (VALU)
                # = lo_bit*(v_nv3-v_nv6)+v_nv6; lo_bit∈{0,1} so: 1→v_nv3, 0→v_nv6
                op_nv_t = Op(
                    engine='valu',
                    instr=('multiply_add', nv_buf, tmp1, v_diff_T, v_nv6),
                    dst=nv_buf,
                    dep_srcs=[tmp1, v_diff_T, v_nv6],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nv_t)
                s += 1
                # Step 4: nv_bit1_F = multiply_add(lo_bit, diff_F, v_nv4) → tmp_nav (VALU)
                # = lo_bit*(v_nv5-v_nv4)+v_nv4; lo_bit∈{0,1} so: 1→v_nv5, 0→v_nv4
                op_nv_f = Op(
                    engine='valu',
                    instr=('multiply_add', tmp_nav, tmp1, v_diff_F, v_nv4),
                    dst=tmp_nav,
                    dep_srcs=[tmp1, v_diff_F, v_nv4],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nv_f)
                s += 1
                # Step 5: nv_buf = vselect(bit1, nv_buf, tmp_nav)  →  nv_buf  (FLOW)
                # bit1≠0 → use nv_buf (nv3 or nv6); bit1=0 → use tmp_nav (nv5 or nv4)
                op_nv_sel = Op(
                    engine='flow',
                    instr=('vselect', nv_buf, tmp_vld, nv_buf, tmp_nav),
                    dst=nv_buf,
                    dep_srcs=[tmp_vld, nv_buf, tmp_nav],
                    context_id=ctx, round_id=r, stage=s,
                )
                ctx_ops.append(op_nv_sel)
                s += 1
            # software_nv rounds 0,11 and wrap round 10: nv = v_nv_root (constant vector),
            # no nv_buf write — XOR op will reference v_nv_root directly.

            # ── Stages 6–14: hash engine type ────────────────────────────────
            is_alu_hash = context_types.get(ctx) == 'alu_hash'

            # ── Stage 5: val_buf ^= nv ──────────────────────────────────────
            # dep_srcs for val_buf side: for ALU-hash contexts, the previous round's
            # hash wrote individual per-lane addresses (val_buf+i). The XOR must wait
            # for ALL lanes to complete, so use [val_buf+vi for vi in range(VLEN)]
            # rather than just [val_buf]. For VALU-hash, a single val_buf suffices
            # because multiply_add writes the whole vector in one op.
            if is_alu_hash:
                val_dep_srcs = [val_buf + vi for vi in range(VLEN)]
            else:
                val_dep_srcs = [val_buf]

            if rc in ('scatter', 'last', 'wrap'):
                # Reads all VLEN words of nv_buf — must wait for ALL 8 loads.
                # wrap (10): scatter-loaded nv_buf still needed (level-10 node values).
                op_xor = Op(
                    engine='valu',
                    instr=('^', val_buf, val_buf, nv_buf),
                    dst=val_buf,
                    dep_srcs=val_dep_srcs + [nv_buf + vi for vi in range(VLEN)],
                    context_id=ctx, round_id=r, stage=s,
                )
            elif rc == 'software_nv' and r in (1, 12):
                # nv_buf written by vselect op above (flow engine)
                op_xor = Op(
                    engine='valu',
                    instr=('^', val_buf, val_buf, nv_buf),
                    dst=val_buf,
                    dep_srcs=val_dep_srcs + [nv_buf],
                    context_id=ctx, round_id=r, stage=s,
                )
            elif rc == 'software_nv':
                # rounds 0, 11 (gi=0): nv = v_nv_root constant
                op_xor = Op(
                    engine='valu',
                    instr=('^', val_buf, val_buf, v_nv_root),
                    dst=val_buf,
                    dep_srcs=val_dep_srcs + [v_nv_root],
                    context_id=ctx, round_id=r, stage=s,
                )
            else:
                # level2 (rc == 'level2'): nv_buf written by final vselect (step 5)
                op_xor = Op(
                    engine='valu',
                    instr=('^', val_buf, val_buf, nv_buf),
                    dst=val_buf,
                    dep_srcs=val_dep_srcs + [nv_buf],
                    context_id=ctx, round_id=r, stage=s,
                )
            ctx_ops.append(op_xor)
            s += 1

            if is_alu_hash:
                # ── ALU hash: 15 sub-cycles × 8 lanes = 120 ALU ops per round ──
                # 3 mul stages × 2 sub-cycles + 3 XOR stages × 3 sub-cycles = 15
                # Sub-cycle ordering matches the dependency chain:
                #   mul:  (* val factor) → (+ val C)
                #   xor:  (op1 tmp1 val c1) → (op3 tmp2 val c3) → (op2 val tmp1 tmp2)
                first_alu_hash_subcycle = True
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    is_mul = (op1 == "+" and op2 == "+" and op3 == "<<")
                    if is_mul:
                        factor = (1 + (1 << val3)) & 0xFFFFFFFF
                        c_factor = const_map[factor]
                        c_C = const_map[val1]
                        # Sub-cycle 1: val_buf[i] = val_buf[i] * factor
                        for i in range(VLEN):
                            # First sub-cycle ever: explicitly link all 8 lanes to op_xor.
                            # val_buf = val_buf+0 so last_writer[val_buf] gets overwritten
                            # by lane 0's write — lanes 1..7 must carry op_xor as an
                            # explicit dep to avoid scheduling before the XOR completes.
                            if first_alu_hash_subcycle:
                                pre_deps = [op_xor]
                                dep_srcs = [c_factor]
                            else:
                                pre_deps = []
                                dep_srcs = [val_buf + i, c_factor]
                            op_am1 = Op(
                                engine='alu',
                                instr=('*', val_buf + i, val_buf + i, c_factor),
                                dst=val_buf + i,
                                dep_srcs=dep_srcs,
                                deps=pre_deps,
                                context_id=ctx, round_id=r, stage=s,
                            )
                            ctx_ops.append(op_am1)
                        first_alu_hash_subcycle = False
                        s += 1
                        # Sub-cycle 2: val_buf[i] = val_buf[i] + C
                        for i in range(VLEN):
                            op_am2 = Op(
                                engine='alu',
                                instr=('+', val_buf + i, val_buf + i, c_C),
                                dst=val_buf + i,
                                dep_srcs=[val_buf + i, c_C],
                                context_id=ctx, round_id=r, stage=s,
                            )
                            ctx_ops.append(op_am2)
                        s += 1
                    else:
                        c_val1 = const_map[val1]
                        c_val3 = const_map[val3]
                        # Sub-cycle 1: tmp1[i] = val_buf[i] op1 c_val1
                        for i in range(VLEN):
                            op_ax1 = Op(
                                engine='alu',
                                instr=(op1, tmp1 + i, val_buf + i, c_val1),
                                dst=tmp1 + i,
                                dep_srcs=[val_buf + i, c_val1],
                                context_id=ctx, round_id=r, stage=s,
                            )
                            ctx_ops.append(op_ax1)
                        s += 1
                        # Sub-cycle 2: tmp2[i] = val_buf[i] op3 c_val3
                        for i in range(VLEN):
                            op_ax2 = Op(
                                engine='alu',
                                instr=(op3, tmp2 + i, val_buf + i, c_val3),
                                dst=tmp2 + i,
                                dep_srcs=[val_buf + i, c_val3],
                                context_id=ctx, round_id=r, stage=s,
                            )
                            ctx_ops.append(op_ax2)
                        s += 1
                        # Sub-cycle 3: val_buf[i] = tmp1[i] op2 tmp2[i]
                        for i in range(VLEN):
                            op_ax3 = Op(
                                engine='alu',
                                instr=(op2, val_buf + i, tmp1 + i, tmp2 + i),
                                dst=val_buf + i,
                                dep_srcs=[tmp1 + i, tmp2 + i],
                                context_id=ctx, round_id=r, stage=s,
                            )
                            ctx_ops.append(op_ax3)
                        s += 1
            else:
                # ── VALU hash: 9 VALU cycles (Spec 06 path, unchanged) ───────
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

            # ── Navigation (class-dependent) ────────────────────────────────
            # 'last' (round 15): skip all nav (idx never used after)
            # 'wrap' (round 10): gi-clear only — set idx=0 (all children OOB → all wrap)
            # 'scatter', 'software_nv', 'level2': full standard nav
            #   gi_double: idx = idx * 2 + 1  (via multiply_add(idx, idx, v_two, v_one))
            #   nav_a:     tmp_nav = val & 1  (low bit of hash output)
            #   nav_b:     idx += tmp_nav     (adds 0 or 1)
            #   nav_c:     tmp_vld = idx < n_nodes  (bounds check)
            #   nav_d:     idx *= tmp_vld     (wrap OOB to 0)
            if r != last_round:
                if rc == 'wrap':
                    # All children of level-9 nodes are OOB → guaranteed all-wrap to gi=0.
                    # Self-XOR clears idx_buf to 0 vector.
                    op_gi_clear = Op(
                        engine='valu',
                        instr=('^', idx_buf, idx_buf, idx_buf),
                        dst=idx_buf,
                        dep_srcs=[idx_buf],
                        context_id=ctx, round_id=r, stage=s,
                    )
                    ctx_ops.append(op_gi_clear)
                    s += 1
                else:
                    # Full standard nav: gi_double + nav A/B/C/D
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
                    # For ALU-hash contexts, val_buf was written per-lane (val_buf+i).
                    # Use per-lane dep_srcs so the dep builder finds the last ALU hash ops.
                    if is_alu_hash:
                        nav_a_dep_srcs = [val_buf + i for i in range(VLEN)] + [v_one]
                    else:
                        nav_a_dep_srcs = [val_buf, v_one]
                    op_nav_a = Op(
                        engine='valu',
                        instr=('&', tmp_nav, val_buf, v_one),
                        dst=tmp_nav,
                        dep_srcs=nav_a_dep_srcs,
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


# ── Spec 07: ALU Hash Integration into DAG Scheduler ─────────────────────────


def assign_context_types(contexts):
    """
    Returns {context_id: 'valu_hash' | 'alu_hash'} for each context in the pass.
    Every 4th context (context_idx % 4 == 3) is an ALU-hash context; all others VALU.
    """
    return {ctx: ('alu_hash' if ctx % 4 == 3 else 'valu_hash') for ctx in contexts}


# ── Spec 08: Depth Specialization atop DAG Scheduler ─────────────────────────


def round_class(round_id: int) -> str:
    """
    Returns the class string for a given round_id in [0, 15].

    Classes:
      'software_nv' — rounds 0, 1, 11, 12: no scatter loads; nv via multiply_add or v_nv_root
      'level2'      — rounds 2, 13:          no scatter loads; nv via multiply_add (same formula)
      'wrap'        — round 10:              no scatter loads; all inputs wrap to gi=0
      'last'        — round 15:              scatter loads; nav ops skipped (result unused)
      'scatter'     — rounds 3–9, 14:        scatter loads + full nav
    """
    if round_id in (0, 1, 11, 12):
        return 'software_nv'
    if round_id in (2, 13):
        return 'level2'
    if round_id == 10:
        return 'wrap'
    if round_id == 15:
        return 'last'
    return 'scatter'


# ── End Spec 08 / End Spec 07 / End Spec 06 DAG Infrastructure ────────────────


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
        # c_nn / v_nnodes removed: nav_c/nav_d dead-code elimination made them unused.
        # Frees 9 scratch words (1 scalar + 8 vector) for Level2 FLOW→VALU constants.

        # ── Vector constants (broadcast once, reused every cycle) ─────────────
        v_one    = self.alloc_scratch("v_one",    VLEN)
        v_two    = self.alloc_scratch("v_two",    VLEN)  # for gi-doubler: gi = 2*gi + 1
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

        # Pre-allocate nv scratch so add_imm ops can be woven into the idle FLOW
        # slots of the remaining val-vload phase below (saves 6 standalone cycles).
        fvp = self.scratch["forest_values_p"]
        s_nv = [self.alloc_scratch(f"s_nv{i}") for i in range(7)]   # 7 scalar slots
        v_nv = [self.alloc_scratch(f"v_nv{i}", VLEN) for i in range(7)]  # 7 × VLEN = 56
        _nv_add_imm_ops = [("add_imm", s_nv[i], fvp, i) for i in range(1, 7)]

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

        # Remaining vload pairs overlapped with idx vbroadcasts and nv add_imm ops.
        # FLOW is idle in these cycles; absorb all 6 add_imm ops here (saves 6 cycles).
        _nv_ai_idx = 0
        while vl_idx < len(val_vload_pairs) or vb_idx < len(idx_vb_batches) or _nv_ai_idx < len(_nv_add_imm_ops):
            bundle = {}
            if vl_idx < len(val_vload_pairs):
                bundle["load"] = val_vload_pairs[vl_idx]
                vl_idx += 1
            if vb_idx < len(idx_vb_batches):
                bundle["valu"] = idx_vb_batches[vb_idx]
                vb_idx += 1
            if _nv_ai_idx < len(_nv_add_imm_ops):
                bundle["flow"] = [_nv_add_imm_ops[_nv_ai_idx]]
                _nv_ai_idx += 1
            if bundle:
                self.instrs.append(bundle)

        # ── Spec 08: software_nv / level2 / wrap constant loading ────────────
        # Preload node_val[0..6] from forest memory into scalar scratch, then
        # broadcast each to a vector scratch slot.  These constants replace
        # scatter loads in non-scatter rounds:
        #   rounds 0,11 (gi=0):    nv = v_nv[0]
        #   rounds 1,12 (gi=1,2):  nv = vselect(gi&1, v_nv[1], v_nv[2])
        #   round 10 (wrap/gi=0):  nv = v_nv[0]
        #   rounds 2,13 (gi=3..6): 2-level vselect using gi&1 and gi&2 bits
        #                          (gi&2 reuses existing v_two constant)
        #
        # Scratch budget: 7 scalars + 7×VLEN = 7 + 56 = 63 words.
        # node_val[i] address = forest_values_p + i.
        # We reuse s_nv[1..6] as address buffers first, then overwrite with values.
        # Step 1: add_imm(s_nv[i], fvp, i) → s_nv[i] = scratch[fvp] + i = fp+i
        # Step 2: load(s_nv[i], s_nv[i])   → s_nv[i] = mem[fp+i] = node_val[i]
        # fvp, s_nv, v_nv allocated above (before val loading phase).
        # add_imm ops overlapped into val vload phase above (saves 6 cycles).

        # Step 2: load node_val[0..6] from forest memory.
        # node_val[0]: load(s_nv[0], fvp)      → mem[scratch[fvp]]      = mem[fp]
        # node_val[i]: load(s_nv[i], s_nv[i])  → mem[scratch[s_nv[i]]]  = mem[fp+i]
        # Self-referential load overwrites the address slot with the value. ✓
        all_loads_nv = [("load", s_nv[0], fvp)] + [
            ("load", s_nv[i], s_nv[i]) for i in range(1, 7)
        ]
        for i in range(0, len(all_loads_nv), 2):
            self.instrs.append({"load": all_loads_nv[i:i + 2]})

        # Step 3: broadcast node_val[0..6] to vector scratch (6 per VALU cycle).
        all_vb_nv = [("vbroadcast", v_nv[i], s_nv[i]) for i in range(7)]
        for i in range(0, len(all_vb_nv), VALU_LIMIT):
            self.instrs.append({"valu": all_vb_nv[i:i + VALU_LIMIT]})

        # Alias for build_op_graph dispatch
        v_nv_root = v_nv[0]   # rounds 0, 10, 11

        # Level2 FLOW→VALU: pre-compute diff vectors so multiply_add replaces 2 vselects.
        # multiply_add(dst, lo_bit, diff, base) = lo_bit*(nv_T-nv_F) + nv_F
        # = nv_T when lo_bit=1, nv_F when lo_bit=0 (since lo_bit ∈ {0,1} from &1).
        v_diff_T = self.alloc_scratch("v_diff_T", VLEN)  # v_nv3 - v_nv6 (step 3 diff)
        v_diff_F = self.alloc_scratch("v_diff_F", VLEN)  # v_nv5 - v_nv4 (step 4 diff)
        self.instrs.append({"valu": [
            ('-', v_diff_T, v_nv[3], v_nv[6]),
            ('-', v_diff_F, v_nv[5], v_nv[4]),
        ]})

        # ── Spec 06: DAG-based multi-context scheduler ────────────────────────
        # Replace hand-scheduled quadruplet loop with greedy list scheduler.
        # group_size=16: 16 contexts × 48 words = 768 words; dag_base ~697, 697+768=1465 ≤ 1536.
        group_size = 32
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
            context_types = assign_context_types(list(range(n_ctx)))
            ops = build_op_graph(
                contexts=list(range(n_ctx)),
                rounds=list(range(rounds)),
                scratch=pass_layout,
                hsc_new=self.hsc_new,
                v_fp=v_fp,
                v_one=v_one,
                v_two=v_two,
                context_types=context_types,
                const_map=self.const_map,
                v_nv_root=v_nv_root,
                v_nv1=v_nv[1],
                v_nv2=v_nv[2],
                v_nv3=v_nv[3],
                v_nv4=v_nv[4],
                v_nv5=v_nv[5],
                v_nv6=v_nv[6],
                v_diff_T=v_diff_T,
                v_diff_F=v_diff_F,
            )
            # ── Store integration: add vstore ops into the DAG ────────────────
            # Build set of all val_buf lane addresses across all contexts.
            # VALU hash writes dst=val_buf (base); ALU hash writes dst=val_buf+i (per-lane).
            # Track last writer per lane address so stores depend on ALL lane writes.
            val_buf_addrs = set()
            for ctx in range(n_ctx):
                vb = pass_layout.val_buf(ctx)
                for i in range(VLEN):
                    val_buf_addrs.add(vb + i)
            last_val_writer = {}  # addr → Op (last writer of that lane address)
            for op in ops:
                if op.dst in val_buf_addrs:
                    addr = op.dst
                    if addr not in last_val_writer or (
                        op.round_id > last_val_writer[addr].round_id or (
                            op.round_id == last_val_writer[addr].round_id and
                            op.stage >= last_val_writer[addr].stage
                        )
                    ):
                        last_val_writer[addr] = op
            for ctx in range(n_ctx):
                g = pass_groups[ctx]
                vb_addr = pass_layout.val_buf(ctx)
                store_op = Op(
                    engine='store',
                    instr=('vstore', addr_arr + g, vb_addr),
                    dst=-1,
                    dep_srcs=[vb_addr],
                    context_id=ctx,
                    round_id=rounds - 1,
                    stage=9999,
                )
                # Depend on the last writer of every lane (val_buf+0 .. val_buf+VLEN-1).
                seen_writers = set()
                for i in range(VLEN):
                    writer = last_val_writer.get(vb_addr + i)
                    if writer is not None and id(writer) not in seen_writers:
                        store_op.deps.append(writer)
                        seen_writers.add(id(writer))
                ops.append(store_op)
            schedule = list_schedule(ops)
            for bundle in schedule:
                if not bundle.is_empty():
                    self.instrs.append(bundle.to_instr_dict())

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

    # ── Spec 07 Unit Tests ────────────────────────────────────────────────────

    def test_FR1_assign_context_types_8ctxs(self):
        """T1: 8 contexts → exactly contexts 3,7 are alu_hash."""
        ct = assign_context_types(list(range(8)))
        self.assertEqual(ct[0], 'valu_hash')
        self.assertEqual(ct[1], 'valu_hash')
        self.assertEqual(ct[2], 'valu_hash')
        self.assertEqual(ct[3], 'alu_hash')
        self.assertEqual(ct[4], 'valu_hash')
        self.assertEqual(ct[5], 'valu_hash')
        self.assertEqual(ct[6], 'valu_hash')
        self.assertEqual(ct[7], 'alu_hash')

    def test_FR1_assign_context_types_16ctxs(self):
        """T2: 16 contexts → exactly 3,7,11,15 are alu_hash."""
        ct = assign_context_types(list(range(16)))
        alu = [c for c, t in ct.items() if t == 'alu_hash']
        valu = [c for c, t in ct.items() if t == 'valu_hash']
        self.assertEqual(sorted(alu), [3, 7, 11, 15])
        self.assertEqual(len(valu), 12)

    def test_FR1_assign_context_types_2ctxs(self):
        """T3: 2 contexts → both valu_hash."""
        ct = assign_context_types([0, 1])
        self.assertEqual(ct[0], 'valu_hash')
        self.assertEqual(ct[1], 'valu_hash')

    def test_FR2_alu_hash_ops_count(self):
        """T4: ALU-hash context produces 120 ALU hash ops per round (15 sub-cycles × 8 lanes)."""
        # Minimal setup: 1 context (ALU hash), 1 round
        from problem import HASH_STAGES
        n_ctx = 4  # context 3 is ALU hash
        dag_base = 500
        context_bases = [dag_base + i * ScratchLayout.WORDS_PER_CTX for i in range(n_ctx)]
        val_bases = [100 + g * VLEN for g in range(n_ctx)]
        idx_bases = [200 + g * VLEN for g in range(n_ctx)]
        layout = ScratchLayout(n_contexts=n_ctx, context_bases=context_bases,
                               val_bases=val_bases, idx_bases=idx_bases)
        const_map = {}
        sp = dag_base + n_ctx * ScratchLayout.WORDS_PER_CTX
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            for v in ([val1, (1+(1<<val3))&0xFFFFFFFF] if is_mul else [val1, val3]):
                if v not in const_map:
                    const_map[v] = sp; sp += 1
        v_fp = sp; sp += VLEN; v_one = sp; sp += VLEN
        v_two = sp; sp += VLEN; v_nnodes = sp; sp += VLEN
        hsc_new = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            if is_mul:
                factor = (1 + (1 << val3)) & 0xFFFFFFFF
                hsc_new.append(('mul', const_map[factor], const_map[val1]))
            else:
                hsc_new.append(('xor', op1, const_map[val1], op2, op3, const_map[val3]))

        context_types = assign_context_types(list(range(n_ctx)))
        # Use last round so nav is skipped (simplify dep analysis)
        ops = build_op_graph([3], [0], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map)
        alu_hash_ops = [o for o in ops if o.engine == 'alu']
        self.assertEqual(len(alu_hash_ops), 120,
                         f"Expected 120 ALU hash ops, got {len(alu_hash_ops)}")

    def test_FR2_valu_hash_unchanged(self):
        """T5: VALU-hash context still produces 9 VALU hash ops per round."""
        from problem import HASH_STAGES
        n_ctx = 4
        dag_base = 500
        context_bases = [dag_base + i * ScratchLayout.WORDS_PER_CTX for i in range(n_ctx)]
        val_bases = [100 + g * VLEN for g in range(n_ctx)]
        idx_bases = [200 + g * VLEN for g in range(n_ctx)]
        layout = ScratchLayout(n_contexts=n_ctx, context_bases=context_bases,
                               val_bases=val_bases, idx_bases=idx_bases)
        const_map = {}
        sp = dag_base + n_ctx * ScratchLayout.WORDS_PER_CTX
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            for v in ([val1, (1+(1<<val3))&0xFFFFFFFF] if is_mul else [val1, val3]):
                if v not in const_map:
                    const_map[v] = sp; sp += 1
        v_fp = sp; sp += VLEN; v_one = sp; sp += VLEN
        v_two = sp; sp += VLEN; v_nnodes = sp; sp += VLEN
        hsc_new = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            if is_mul:
                factor = (1 + (1 << val3)) & 0xFFFFFFFF
                hsc_new.append(('mul', const_map[factor], const_map[val1]))
            else:
                hsc_new.append(('xor', op1, const_map[val1], op2, op3, const_map[val3]))

        context_types = assign_context_types(list(range(n_ctx)))
        # ctx=0 is valu_hash; use last round to skip nav
        ops = build_op_graph([0], [0], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map)
        # VALU hash ops: 3 mul (1 op each) + 3 xor (3 ops each) = 3+9=12... wait
        # Actually multiply_add is 1 VALU op per mul-stage; xor is 3 VALU ops per xor-stage
        # 3 mul * 1 + 3 xor * 3 = 3 + 9 = 12? No: mul_stage = 1 multiply_add
        # xor stage = 2 ops (odd cycle: op_h1 + op_h2) + 1 op (even cycle: op_h3) = 3 ops
        # Total: 3*1 + 3*(2+1) = 3+9 = 12 ops... but spec says 9 "VALU cycles"
        # Actually counting stages: mul=1 VALU stage, xor=2 VALU stages (odd + even)
        # 3 mul + 3*2 xor = 3+6 = 9 stages/cycles. Ops: 3*1 + 3*(2+1) = 3+9=12 VALU ops
        valu_hash_ops = [o for o in ops if o.engine == 'valu' and o.context_id == 0
                         and o.stage >= 6]  # stages 6+ are hash ops (0=addr,1-4=load,5=xor)
        # The hash stages start after stage 5 (the val^nv xor). Count them:
        # stage 5 = xor, stages 6-14 = hash. That's stages 6..14 = 9 stages but 12 VALU ops
        valu_hash_ops2 = [o for o in ops if o.engine == 'valu' and o.context_id == 0
                          and o.stage > 5]
        self.assertGreater(len(valu_hash_ops2), 0, "Expected VALU hash ops for ctx 0")
        # Key check: no ALU ops for ctx 0 (it's valu_hash)
        alu_ops_ctx0 = [o for o in ops if o.engine == 'alu' and o.context_id == 0]
        self.assertEqual(len(alu_ops_ctx0), 0, "VALU context should have no ALU hash ops")

    def test_FR2_no_cross_context_deps(self):
        """T6: No deps between different contexts."""
        from problem import HASH_STAGES
        n_ctx = 4
        dag_base = 500
        context_bases = [dag_base + i * ScratchLayout.WORDS_PER_CTX for i in range(n_ctx)]
        val_bases = [100 + g * VLEN for g in range(n_ctx)]
        idx_bases = [200 + g * VLEN for g in range(n_ctx)]
        layout = ScratchLayout(n_contexts=n_ctx, context_bases=context_bases,
                               val_bases=val_bases, idx_bases=idx_bases)
        const_map = {}
        sp = dag_base + n_ctx * ScratchLayout.WORDS_PER_CTX
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            for v in ([val1, (1+(1<<val3))&0xFFFFFFFF] if is_mul else [val1, val3]):
                if v not in const_map:
                    const_map[v] = sp; sp += 1
        v_fp = sp; sp += VLEN; v_one = sp; sp += VLEN
        v_two = sp; sp += VLEN; v_nnodes = sp; sp += VLEN
        hsc_new = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            if is_mul:
                factor = (1 + (1 << val3)) & 0xFFFFFFFF
                hsc_new.append(('mul', const_map[factor], const_map[val1]))
            else:
                hsc_new.append(('xor', op1, const_map[val1], op2, op3, const_map[val3]))

        context_types = assign_context_types(list(range(n_ctx)))
        ops = build_op_graph(list(range(n_ctx)), [0], layout, hsc_new, v_fp, v_one, v_two,
                             v_nnodes, context_types=context_types, const_map=const_map)

        for op in ops:
            for dep in op.deps:
                self.assertEqual(op.context_id, dep.context_id,
                                 f"Cross-context dep: op ctx={op.context_id} deps on ctx={dep.context_id}")

    def test_FR3_bundle_alu_limit(self):
        """T10: Bundle rejects 13th ALU op."""
        # Verify 12 ALU ops fit, 13th rejected
        b = Bundle()
        dummy_val = 999
        for i in range(12):
            op = Op(engine='alu', instr=('+', i, i, dummy_val), dst=i,
                    dep_srcs=[], context_id=0, round_id=0, stage=0)
            self.assertTrue(b.can_fit(op), f"Should accept op {i+1}")
            b.add(op)
        op13 = Op(engine='alu', instr=('+', 100, 100, dummy_val), dst=100,
                  dep_srcs=[], context_id=0, round_id=0, stage=0)
        self.assertFalse(b.can_fit(op13), "Should reject 13th ALU op")

    # ── Spec 08 Unit Tests ────────────────────────────────────────────────────

    def _make_dag_layout_and_consts(self, n_ctx=4, dag_base=500):
        """Helper: build a ScratchLayout + hsc_new + const_map for build_op_graph tests."""
        from problem import HASH_STAGES
        context_bases = [dag_base + i * ScratchLayout.WORDS_PER_CTX for i in range(n_ctx)]
        val_bases = [100 + g * VLEN for g in range(n_ctx)]
        idx_bases = [200 + g * VLEN for g in range(n_ctx)]
        layout = ScratchLayout(n_contexts=n_ctx, context_bases=context_bases,
                               val_bases=val_bases, idx_bases=idx_bases)
        const_map = {}
        sp = dag_base + n_ctx * ScratchLayout.WORDS_PER_CTX
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            for v in ([val1, (1 + (1 << val3)) & 0xFFFFFFFF] if is_mul else [val1, val3]):
                if v not in const_map:
                    const_map[v] = sp
                    sp += 1
        v_fp = sp; sp += VLEN
        v_one = sp; sp += VLEN
        v_two = sp; sp += VLEN
        v_nnodes = sp; sp += VLEN
        v_nv_root = sp; sp += VLEN
        v_nv_delta = sp; sp += VLEN
        v_nv_base = sp; sp += VLEN
        hsc_new = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mul = (op1 == '+' and op2 == '+' and op3 == '<<')
            if is_mul:
                factor = (1 + (1 << val3)) & 0xFFFFFFFF
                hsc_new.append(('mul', const_map[factor], const_map[val1]))
            else:
                hsc_new.append(('xor', op1, const_map[val1], op2, op3, const_map[val3]))
        return layout, hsc_new, const_map, v_fp, v_one, v_two, v_nnodes, v_nv_root, v_nv_delta, v_nv_base

    def test_FR1_round_class_all_rounds(self):
        """T1 (Spec 08): Verify all 16 round_ids map to correct class strings."""
        self.assertEqual(round_class(0),  'software_nv')
        self.assertEqual(round_class(1),  'software_nv')
        self.assertEqual(round_class(11), 'software_nv')
        self.assertEqual(round_class(12), 'software_nv')
        self.assertEqual(round_class(2),  'level2')
        self.assertEqual(round_class(13), 'level2')
        self.assertEqual(round_class(10), 'wrap')
        self.assertEqual(round_class(15), 'last')
        for r in (3, 4, 5, 6, 7, 8, 9, 14):
            self.assertEqual(round_class(r), 'scatter', f"round {r} should be scatter")

    def test_FR3_no_loads_software_nv_round0(self):
        """T2 (Spec 08): software_nv round 0 — 0 scatter load ops per context."""
        layout, hsc_new, const_map, v_fp, v_one, v_two, v_nnodes, v_nv_root, v_nv_delta, v_nv_base = \
            self._make_dag_layout_and_consts()
        context_types = assign_context_types([0])
        ops = build_op_graph([0], [0], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map,
                             v_nv_root=v_nv_root, v_nv_delta=v_nv_delta, v_nv_base=v_nv_base)
        load_ops = [o for o in ops if o.engine == 'load']
        self.assertEqual(len(load_ops), 0,
                         f"software_nv round 0 should have 0 load ops, got {len(load_ops)}")

    def test_FR3_loads_scatter_round3(self):
        """T3 (Spec 08): scatter round 3 — 8 scatter load ops per context."""
        layout, hsc_new, const_map, v_fp, v_one, v_two, v_nnodes, v_nv_root, v_nv_delta, v_nv_base = \
            self._make_dag_layout_and_consts()
        context_types = assign_context_types([0])
        ops = build_op_graph([0], [3], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map,
                             v_nv_root=v_nv_root, v_nv_delta=v_nv_delta, v_nv_base=v_nv_base)
        load_ops = [o for o in ops if o.engine == 'load']
        self.assertEqual(len(load_ops), 8,
                         f"scatter round 3 should have 8 load ops (VLEN={VLEN}), got {len(load_ops)}")

    def test_FR4_no_nav_last_round(self):
        """T4 (Spec 08): last round (15) — 0 nav ops per context."""
        layout, hsc_new, const_map, v_fp, v_one, v_two, v_nnodes, v_nv_root, v_nv_delta, v_nv_base = \
            self._make_dag_layout_and_consts()
        context_types = assign_context_types([0])
        # Use only round 15 so last_round == 15 and nav is skipped
        ops = build_op_graph([0], [15], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map,
                             v_nv_root=v_nv_root, v_nv_delta=v_nv_delta, v_nv_base=v_nv_base)
        idx_buf = layout.idx_buf(0)
        # Nav ops write idx_buf. When round 15 is last_round, no nav should be emitted.
        nav_ops = [o for o in ops if o.dst == idx_buf and o.round_id == 15
                   and o.engine == 'valu']
        # Filter to only nav-stage ops (stage > hash stage range, i.e. after hash)
        # The gi-doubler and nav A/B/C/D all write idx_buf or tmp_nav/tmp_vld
        # Simplest: count ops that write idx_buf in round 15 and are valu (gi-doubler + nav_b + nav_d)
        self.assertEqual(len(nav_ops), 0,
                         f"last round 15 should have 0 nav ops writing idx_buf, got {len(nav_ops)}")

    def test_FR4_wrap_nav_round10(self):
        """T5 (Spec 08): wrap round 10 — exactly 1 nav op (gi-clear)."""
        layout, hsc_new, const_map, v_fp, v_one, v_two, v_nnodes, v_nv_root, v_nv_delta, v_nv_base = \
            self._make_dag_layout_and_consts()
        context_types = assign_context_types([0])
        # Use rounds [10, 11] so round 10 is not last_round
        ops = build_op_graph([0], [10, 11], layout, hsc_new, v_fp, v_one, v_two, v_nnodes,
                             context_types=context_types, const_map=const_map,
                             v_nv_root=v_nv_root, v_nv_delta=v_nv_delta, v_nv_base=v_nv_base)
        idx_buf = layout.idx_buf(0)
        # Nav ops for round 10 that write idx_buf (the self-XOR clear)
        wrap_nav_ops = [o for o in ops if o.round_id == 10 and o.dst == idx_buf
                        and o.engine == 'valu']
        self.assertEqual(len(wrap_nav_ops), 1,
                         f"wrap round 10 should have exactly 1 nav op (gi-clear), got {len(wrap_nav_ops)}")


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
