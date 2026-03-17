"""
Tests for Spec 03: Navigation Compression
  - FR1: vselect → multiply trick (pure valu navigation)
  - FR2: Multi-group navigation packing (≤ 5 nav cycles per triplet)
  - FR3: Cross-triplet xor+addr overlap into navigation tail

Phase 1.0 — Red phase (tests should fail before implementation)

Run with: python3.11 test_03_scratchpad_cache.py
"""
import random
import unittest

from problem import VLEN, Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
from perf_takehome import KernelBuilder


def _run_kernel(forest_height, batch_size, rounds, seed=123):
    """Build and run the kernel. Returns (cycle_count, kb). Raises if output != reference."""
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace)

    for ref_mem in reference_kernel2(mem, value_trace):
        machine.run()

    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + batch_size]
        == ref_mem[inp_values_p : inp_values_p + batch_size]
    ), "Final val_arr mismatch vs reference"

    return machine.cycle, kb


def _count_flow_vselect(kb):
    """Count all flow:vselect instructions in the kernel."""
    count = 0
    for instr in kb.instrs:
        if "flow" in instr:
            count += sum(1 for slot in instr["flow"] if slot[0] == "vselect")
    return count


def _count_valu_multiply_nav(kb):
    """
    Count valu:* instructions that look like navigation multiply: ("*", dest, dest, src)
    where dest == op[1] == op[2] (gi = gi * tmp_vld pattern).
    """
    count = 0
    for instr in kb.instrs:
        if "valu" in instr:
            for slot in instr["valu"]:
                if slot[0] == "*" and slot[1] == slot[2]:
                    count += 1
    return count


def _is_xor_nv_bundle(instr):
    """
    Identify xor-with-node_value bundles.
    These have at least one valu op where op[0]=="^" AND op[1]==op[2] (dest==src1).
    Bundle must be valu-only (no load/flow/alu).
    """
    if "load" in instr or "flow" in instr or "alu" in instr:
        return False
    if "valu" not in instr:
        return False
    return any(op[0] == "^" and op[1] == op[2] for op in instr["valu"])


def _count_bundles_after_pause(kb):
    """
    Returns the instruction list starting after the first 'pause' instruction.
    The main loop begins after the pause.
    """
    in_main_loop = False
    main_loop_instrs = []
    for instr in kb.instrs:
        if in_main_loop:
            main_loop_instrs.append(instr)
        elif "flow" in instr and any(s[0] == "pause" for s in instr["flow"]):
            in_main_loop = True
    return main_loop_instrs


def _count_flow_vselect_in_main_loop(kb):
    """Count flow:vselect in the main kernel loop (after setup pause)."""
    count = 0
    for instr in _count_bundles_after_pause(kb):
        if "flow" in instr:
            count += sum(1 for slot in instr["flow"] if slot[0] == "vselect")
    return count


class TestSpec03NavigationCompression(unittest.TestCase):

    # ── FR1: vselect → multiply ───────────────────────────────────────────────

    def test_FR1_no_flow_vselect_in_main_loop(self):
        """FR1: Zero flow:vselect instructions must exist in the main kernel loop."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        count = _count_flow_vselect_in_main_loop(kb)
        self.assertEqual(count, 0,
            f"Found {count} flow:vselect instructions in main loop. "
            "These must be replaced with valu:* (multiply) instructions.")

    def test_FR1_valu_multiply_nav_present(self):
        """FR1: valu:* navigation multiply instructions must be present."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        count = _count_valu_multiply_nav(kb)
        # 32 groups per round × 1 round = 32 multiply ops minimum
        self.assertGreaterEqual(count, 32,
            f"Expected ≥ 32 valu:* nav multiply ops, got {count}. "
            "Each group needs gi = gi * tmp_vld.")

    def test_FR1_multiply_count_matches_groups(self):
        """FR1: Number of nav multiply ops should be exactly n_groups per round."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1
        count = _count_valu_multiply_nav(kb)
        # 32 groups × 1 round = 32; with packing this is still 32 multiply operations
        self.assertEqual(count, 32,
            f"Expected 32 valu:* nav multiply ops (32 groups × 1 round), got {count}.")

    def test_FR1_correctness_after_multiply_nav(self):
        """FR1: Kernel output must match reference_kernel2 after vselect→multiply change."""
        try:
            cycles, _ = _run_kernel(10, 256, 16, seed=123)
        except AssertionError as e:
            self.fail(f"Correctness check failed: {e}")
        self.assertGreater(cycles, 0, "Machine ran 0 cycles")

    def test_FR1_correctness_multiple_seeds(self):
        """FR1: Correctness must hold across different random seeds."""
        for seed in [42, 99, 777, 1337, 9999]:
            with self.subTest(seed=seed):
                try:
                    _run_kernel(10, 256, 16, seed=seed)
                except AssertionError as e:
                    self.fail(f"Correctness failed for seed={seed}: {e}")

    def test_FR1_cycle_count_improves_vs_spec02(self):
        """FR1: Cycle count must improve vs spec 02 (5,316 cycles)."""
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLess(cycles, 5316,
            f"Expected < 5,316 cycles after FR1 (vselect→multiply), got {cycles}.")

    # ── FR2: Per-group scratch allocation ─────────────────────────────────────

    def test_FR2_per_group_tmp_nav_scratch_allocated(self):
        """FR2: tmp_nav_A, tmp_nav_B, tmp_nav_C must exist in scratch."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        for name in ("tmp_nav_A", "tmp_nav_B", "tmp_nav_C"):
            self.assertIn(name, kb.scratch,
                f"'{name}' not in scratch. FR2 requires per-group nav temporaries.")

    def test_FR2_per_group_tmp_vld_scratch_allocated(self):
        """FR2: tmp_vld_A, tmp_vld_B, tmp_vld_C must exist in scratch."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        for name in ("tmp_vld_A", "tmp_vld_B", "tmp_vld_C"):
            self.assertIn(name, kb.scratch,
                f"'{name}' not in scratch. FR2 requires per-group vld temporaries.")

    def test_FR2_old_shared_tmp_nav_removed(self):
        """FR2: Old shared 'tmp_nav' and 'tmp_vld' must be removed (replaced by A/B/C variants)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        # The old single shared tmp_nav/tmp_vld must not exist any more
        self.assertNotIn("tmp_nav", kb.scratch,
            "Old shared 'tmp_nav' still in scratch — must be replaced by tmp_nav_A/B/C.")
        self.assertNotIn("tmp_vld", kb.scratch,
            "Old shared 'tmp_vld' still in scratch — must be replaced by tmp_vld_A/B/C.")

    def test_FR2_per_group_tmp_regions_distinct(self):
        """FR2: All 6 per-group nav temp regions must have distinct base addresses."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        names = ("tmp_nav_A", "tmp_nav_B", "tmp_nav_C",
                 "tmp_vld_A", "tmp_vld_B", "tmp_vld_C")
        if not all(n in kb.scratch for n in names):
            self.skipTest("Per-group nav temps not yet allocated")
        addrs = [kb.scratch[n] for n in names]
        self.assertEqual(len(set(addrs)), 6,
            f"Duplicate scratch addresses among per-group nav temps: {list(zip(names, addrs))}")

    # ── FR2: Navigation packing — ≤ 5 nav cycles per triplet ─────────────────

    def test_FR2_nav_cycles_packed_for_full_triplet(self):
        """
        FR2: Navigation for a full triplet (3 groups) must use ≤ 5 nav cycles total.
        This is verified by checking that valu:* multiply ops are packed (3 per cycle
        for a full triplet, not 1 per cycle × 3 groups).
        """
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        # Count cycles with exactly 3 nav multiply ops (full triplet packed)
        # Full triplet packed: 1 cycle contains 3 `("*", gi_gX, gi_gX, tmp_vld_X)` ops
        three_mul_cycles = sum(
            1 for instr in kb.instrs
            if "valu" in instr
            and sum(1 for s in instr["valu"] if s[0] == "*" and s[1] == s[2]) == 3
        )
        # 10 full triplets × 1 nav-E cycle = 10 three-mul cycles per round
        self.assertEqual(three_mul_cycles, 10,
            f"Expected 10 nav-E cycles with 3 packed multiply ops (full triplets), got {three_mul_cycles}. "
            "Navigation must be packed across all groups in a triplet.")

    def test_FR2_nav_step_B_packs_6_ops(self):
        """
        FR2: Navigation step B (gi += gi; tmp_nav += v_one) for full triplet must be
        packed into 1 cycle with 6 valu ops.
        """
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        # After FR3, nav step B bundles may also contain xor/addr ops from next triplet.
        # However, step B ITSELF should always have exactly 6 valu ops for full triplets
        # since it fills all 6 slots.
        # We look for bundles within nav sequence that have exactly 6 valu ops
        # and contain at least one `("+", gi, gi, gi)` pattern (gi+=gi, doubling).
        gi_double_6_cycles = sum(
            1 for instr in kb.instrs
            if "valu" in instr
            and len(instr["valu"]) == 6
            and any(s[0] == "+" and s[1] == s[2] == s[3] for s in instr["valu"])
        )
        # 10 full triplets × 1 nav-B cycle = 10 such cycles per round
        # (partial triplet nav-B has 4 ops, not 6)
        self.assertEqual(gi_double_6_cycles, 10,
            f"Expected 10 nav-B cycles with 6 valu ops (full triplet), got {gi_double_6_cycles}.")

    # ── FR3: Cross-triplet xor+addr overlap ───────────────────────────────────

    def test_FR3_no_standalone_xor_addr_bundles_for_nonfirst_triplets(self):
        """
        FR3: Non-last, non-first triplets must not have standalone xor+addr bundles.

        With spec 02 (no FR3), there are 11 xor+addr bundles per round, each appearing
        as a separate instruction bundle BEFORE the hash phase of each triplet.
        With FR3, these xor/addr ops are folded into the nav tail of the PRIOR triplet,
        so no standalone xor+addr bundles exist for non-last triplets.

        Standalone xor+addr bundle: a bundle that contains ONLY valu ops where
        at least one is "^" (xor) and optionally some "+" (addr_next), with no
        load/flow ops. These are the spec-02-style pre-hash bundles.

        With FR3: triplet-1 through triplet-10 (non-last) should have no standalone
        xor+addr bundle. Only the last triplet or the first triplet may have one.
        Count should be ≤ 2 (first round prologue + last triplet).
        """
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        # Count standalone xor+addr bundles (those are only valu ops, include xor + possibly addr)
        standalone_xor_bundles = [
            instr for instr in _count_bundles_after_pause(kb)
            if _is_xor_nv_bundle(instr)
            and "load" not in instr
        ]

        # With FR3: only the last triplet has a standalone xor bundle (no nav tail to fold into)
        # The first triplet's xor is folded into the prologue structure
        # So expect ≤ 1 standalone xor bundle per round (last triplet only)
        self.assertLessEqual(len(standalone_xor_bundles), 1,
            f"Expected ≤ 1 standalone xor bundle per round (last triplet only), "
            f"got {len(standalone_xor_bundles)}. "
            "Non-last triplet xor+addr ops must be packed into nav tail cycles. "
            "Current spec-02 has 11 such bundles — FR3 eliminates 10 of them.")

    def test_FR3_xor_packed_into_nav_tail(self):
        """
        FR3: XOR ops for non-last triplets must appear in bundles that also contain
        navigation operations (nav-C: gi += tmp_nav pattern).
        """
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        # nav-C bundle pattern: valu contains "+" ops AND "^" ops AND "*" or "<" ops (nav ops)
        # OR more simply: contains xor ops AND navigation-step ops (not just hash ops)
        # Nav ops: any "+" where op is not a hash stage op (hard to distinguish)
        # Simpler heuristic: bundles with xor ops AND also multiply ops (nav-E) or
        # less-than ops (nav-D) co-located = means xor+nav overlap

        # Count bundles that contain xor (val ^= nv) AND also "+" with dest!=src (nav step C)
        nav_c_with_xor = sum(
            1 for instr in _count_bundles_after_pause(kb)
            if "valu" in instr
            and "load" not in instr
            and any(op[0] == "^" and op[1] == op[2] for op in instr["valu"])  # xor nv
            and any(op[0] == "+" for op in instr["valu"])  # some + op (could be nav step C)
        )
        # 10 non-last full triplets: each has xor folded into nav-C
        # = 10 such bundles per round (last triplet has standalone xor)
        self.assertGreaterEqual(nav_c_with_xor, 9,
            f"Expected ≥ 9 nav-C bundles with packed xor ops, got {nav_c_with_xor}. "
            "This means xor is not being folded into navigation tail cycles.")

    def test_FR3_cycle_count_improves_vs_fr1_fr2(self):
        """
        FR3: Cycle count must be lower than what FR1+FR2 alone would achieve.
        Target: < 4,800 cycles (FR1+FR2 eliminates flow bottleneck, FR3 removes xor bundles).
        """
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLess(cycles, 4800,
            f"Expected < 4,800 cycles after FR1+FR2+FR3, got {cycles}.")

    def test_FR3_correctness_maintained(self):
        """FR3: Kernel must still produce correct output after xor pipelining."""
        for seed in [123, 42, 999]:
            with self.subTest(seed=seed):
                try:
                    _run_kernel(10, 256, 16, seed=seed)
                except AssertionError as e:
                    self.fail(f"Correctness failed for seed={seed}: {e}")

    # ── Combined performance target ───────────────────────────────────────────

    def test_combined_cycle_count_target(self):
        """
        Combined FR1+FR2+FR3: Cycle count must be ≤ 2,664
        (11 triplets × 16 cycles × 16 rounds + ~200 overhead).
        """
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLessEqual(cycles, 2664,
            f"Expected ≤ 2,664 cycles (spec 03 target), got {cycles}.")

    def test_beats_opus4_many_hours_threshold(self):
        """Performance: Must beat Claude Opus 4 many-hours threshold (< 2,164 cycles)."""
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLess(cycles, 2164,
            f"Expected < 2,164 cycles (Opus 4 many-hours threshold), got {cycles}.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
