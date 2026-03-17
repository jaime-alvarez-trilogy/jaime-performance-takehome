"""
Tests for Spec 02: Load/Compute Overlap (Software Pipeline)
Phase 1.0 — Red phase (all tests should fail before implementation)

Run with: python3.11 test_02_load_hide.py
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


def _is_xor_nv_bundle(instr):
    """
    Identify xor-with-node_value bundles.
    These have at least one valu op where op[0]=="^" AND op[1]==op[2] (dest==src1).
    Hash "^" ops always have dest != src1, so this uniquely identifies gv ^= nv.
    Bundle must be valu-only (no load/flow/alu).
    """
    if "load" in instr or "flow" in instr or "alu" in instr:
        return False
    if "valu" not in instr:
        return False
    return any(op[0] == "^" and op[1] == op[2] for op in instr["valu"])


def _is_load_offset_only_bundle(instr):
    """True if bundle contains only load_offset ops (no valu/alu/flow)."""
    if "valu" in instr or "alu" in instr or "flow" in instr:
        return False
    return "load" in instr and all(op[0] == "load_offset" for op in instr["load"])


def _is_mixed_valu_load_offset_bundle(instr):
    """True if bundle has both valu ops and load_offset ops."""
    return (
        "valu" in instr
        and "load" in instr
        and all(op[0] == "load_offset" for op in instr["load"])
    )


class TestSpec02LoadHide(unittest.TestCase):

    # ── FR1: Double-Buffer Scratch Allocation ─────────────────────────────────

    def test_FR1_new_regions_in_scratch(self):
        """FR1: nv_A, nv_B, addr_next must all be present in scratch."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        for name in ("nv_A", "nv_B", "addr_next"):
            self.assertIn(name, kb.scratch, f"'{name}' not in scratch")

    def test_FR1_old_nv_and_addr_removed(self):
        """FR1: 'nv' and 'addr' scratch regions must be removed."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        self.assertNotIn("nv", kb.scratch, "'nv' still in scratch — must be removed")
        self.assertNotIn("addr", kb.scratch, "'addr' still in scratch — must be removed")

    def test_FR1_new_regions_do_not_overlap(self):
        """FR1: nv_A, nv_B, addr_next must each span VLEN*3=24 words without overlap."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        names = ("nv_A", "nv_B", "addr_next")
        if not all(n in kb.scratch for n in names):
            self.skipTest("Regions not yet allocated")
        addrs = sorted((kb.scratch[n], n) for n in names)
        for i in range(len(addrs) - 1):
            gap = addrs[i + 1][0] - addrs[i][0]
            self.assertGreaterEqual(
                gap, VLEN * 3,
                f"{addrs[i][1]} and {addrs[i+1][1]}: gap={gap} < VLEN*3={VLEN*3}")

    def test_FR1_all_three_regions_distinct(self):
        """FR1: nv_A, nv_B, addr_next must have distinct base addresses."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)
        names = ("nv_A", "nv_B", "addr_next")
        if not all(n in kb.scratch for n in names):
            self.skipTest("Regions not yet allocated")
        addrs = [kb.scratch[n] for n in names]
        self.assertEqual(len(set(addrs)), 3,
                         f"Duplicate addresses: {list(zip(names, addrs))}")

    # ── FR2: Prologue Load Coverage ───────────────────────────────────────────

    def test_FR2_prologue_emits_24_load_offset_ops(self):
        """FR2: Prologue must emit 24 load_offset ops per round (3 groups × VLEN=8)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        load_only_bundles = [
            instr for instr in kb.instrs if _is_load_offset_only_bundle(instr)
        ]
        total_ops = sum(len(b["load"]) for b in load_only_bundles)
        self.assertEqual(total_ops, 24,
            f"Expected 24 prologue load_offset ops (rounds=1), got {total_ops}")

    def test_FR2_prologue_bundles_each_have_2_loads(self):
        """FR2: Each prologue load bundle must have exactly 2 load_offset ops."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        load_only_bundles = [
            instr for instr in kb.instrs if _is_load_offset_only_bundle(instr)
        ]
        for i, bundle in enumerate(load_only_bundles):
            self.assertEqual(len(bundle["load"]), 2,
                f"Prologue load bundle #{i} has {len(bundle['load'])} ops, expected 2")

    # ── FR3: Xor+Addr Bundle Packing ─────────────────────────────────────────

    def test_FR3_total_xor_bundles_per_round(self):
        """FR3: Must have exactly 11 xor bundles per round for n_triplets=11."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        xor_bundles = [instr for instr in kb.instrs if _is_xor_nv_bundle(instr)]
        self.assertEqual(len(xor_bundles), 11,
            f"Expected 11 xor bundles (11 triplets × 1 round), got {len(xor_bundles)}")

    def test_FR3_full_nolast_triplets_pack_6_ops(self):
        """FR3: Triplets 0-8 (full, non-last, next=full) must have 6-op xor+addr bundles."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        xor_bundles = [instr for instr in kb.instrs if _is_xor_nv_bundle(instr)]
        count_6 = sum(1 for b in xor_bundles if len(b["valu"]) == 6)
        self.assertEqual(count_6, 9,
            f"Expected 9 six-op xor+addr bundles (triplets 0-8), got {count_6}")

    def test_FR3_triplet9_packs_5_ops(self):
        """FR3: Triplet 9 (full, non-last, next=partial 2 groups) must have 5-op xor+addr bundle."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        xor_bundles = [instr for instr in kb.instrs if _is_xor_nv_bundle(instr)]
        count_5 = sum(1 for b in xor_bundles if len(b["valu"]) == 5)
        self.assertEqual(count_5, 1,
            f"Expected 1 five-op xor+addr bundle (triplet 9, next has 2 groups), got {count_5}")

    def test_FR3_last_triplet_xor_only(self):
        """FR3: Last triplet (partial, 2 groups) xor bundle must have no addr_next ops."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        xor_bundles = [instr for instr in kb.instrs if _is_xor_nv_bundle(instr)]
        # Triplet 10: 2 groups → 2 xor ops, no addr_next
        count_small = sum(1 for b in xor_bundles if len(b["valu"]) <= 3)
        self.assertEqual(count_small, 1,
            f"Expected 1 last-triplet xor-only bundle (2 ops), got {count_small}")

    # ── FR4: Hash Prefetch Loads ──────────────────────────────────────────────

    def test_FR4_emit_triplet_hash_with_prefetch_exists(self):
        """FR4: KernelBuilder must have emit_triplet_hash_with_prefetch method."""
        kb = KernelBuilder()
        self.assertTrue(hasattr(kb, "emit_triplet_hash_with_prefetch"),
                        "emit_triplet_hash_with_prefetch not found on KernelBuilder")
        self.assertTrue(callable(getattr(kb, "emit_triplet_hash_with_prefetch")),
                        "emit_triplet_hash_with_prefetch is not callable")

    def test_FR4_mixed_bundle_count(self):
        """FR4: Must have 116 mixed valu+load bundles per round (non-last triplets' hash)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # rounds=1

        mixed = [instr for instr in kb.instrs if _is_mixed_valu_load_offset_bundle(instr)]
        # Triplets 0-8: 9 × 12 = 108 bundles (all 12 cycles have 2 loads, n_next=3)
        # Triplet 9: 8 bundles (first 8 of 12 cycles have 2 loads, n_next=2 → 16 loads)
        # Triplet 10: 0 (last triplet, no prefetch)
        self.assertEqual(len(mixed), 116,
            f"Expected 116 mixed valu+load bundles (rounds=1), got {len(mixed)}")

    def test_FR4_mixed_bundles_each_have_2_loads(self):
        """FR4: Each mixed hash+prefetch bundle must have exactly 2 load_offset ops."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        mixed = [instr for instr in kb.instrs if _is_mixed_valu_load_offset_bundle(instr)]
        for i, bundle in enumerate(mixed):
            self.assertEqual(len(bundle["load"]), 2,
                f"Mixed bundle #{i} has {len(bundle['load'])} load ops, expected 2")

    def test_FR4_last_triplet_hash_has_no_loads(self):
        """FR4: Last triplet (partial, 2 groups) hash must be valu-only (no prefetch)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        # Last triplet hash C1 bundles: 4 valu ops, no load (2 groups × 2 ops)
        # Nav bundles don't produce 4-op bundles, so this is unique.
        valu_only_4 = [
            instr for instr in kb.instrs
            if instr.get("valu") and len(instr["valu"]) == 4
            and "load" not in instr and "flow" not in instr
        ]
        # 6 hash stages × 1 C1 bundle each = 6 four-valu-no-load bundles per round
        self.assertEqual(len(valu_only_4), 6,
            f"Expected 6 four-valu-no-load bundles (last triplet hash C1), got {len(valu_only_4)}")

    # ── FR5: Correctness ─────────────────────────────────────────────────────

    def test_FR5_correctness_matches_reference(self):
        """FR5: Kernel must produce same val output as reference_kernel2 for all rounds."""
        try:
            cycles, _ = _run_kernel(10, 256, 16, seed=123)
        except AssertionError as e:
            self.fail(f"Correctness check failed: {e}")
        self.assertGreater(cycles, 0, "Machine ran 0 cycles")

    def test_FR5_correctness_multiple_seeds(self):
        """FR5: Correctness must hold across different random seeds."""
        for seed in [42, 99, 777]:
            with self.subTest(seed=seed):
                try:
                    _run_kernel(10, 256, 16, seed=seed)
                except AssertionError as e:
                    self.fail(f"Correctness failed for seed={seed}: {e}")

    # ── FR6: Performance ─────────────────────────────────────────────────────

    def test_FR6_cycle_count_below_5600(self):
        """FR6: Total cycle count must be ≤ 5,600 (spec 01: ~8,004)."""
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLessEqual(cycles, 5600,
            f"Expected ≤ 5,600 cycles after spec 02, got {cycles}. "
            f"Spec 01 baseline was ~8,004 cycles.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
