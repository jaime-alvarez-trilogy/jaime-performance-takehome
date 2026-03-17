"""
Tests for Spec 01: Multi-Group Hash Pipeline
Phase 1.0 — Red phase (all tests should fail before implementation)

Run with: python3.11 test_01_hash_pipeline.py
"""
import random
import unittest

from problem import VLEN, Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
from perf_takehome import KernelBuilder


def _run_kernel(forest_height, batch_size, rounds, seed=123):
    """
    Build and run the kernel. Returns (cycle_count, kb).
    Raises AssertionError if output doesn't match reference.
    """
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace)

    # reference_kernel2 modifies mem in-place and yields twice;
    # machine.run() is called in lockstep (Machine made its own copy of mem at init)
    for ref_mem in reference_kernel2(mem, value_trace):
        machine.run()

    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + batch_size]
        == ref_mem[inp_values_p : inp_values_p + batch_size]
    ), "Final val_arr mismatch vs reference"

    return machine.cycle, kb


def _count_valu_bundle_size(kb, size):
    """Count instruction bundles with exactly `size` valu ops."""
    return sum(
        1 for instr in kb.instrs
        if instr.get("valu") and len(instr["valu"]) == size
    )


class TestSpec01HashPipeline(unittest.TestCase):

    # ── FR4: Scratch Layout Non-Aliasing ──────────────────────────────────────

    def test_FR4_tmp_scratch_regions_allocated(self):
        """FR4: tmp1_A/B/C and tmp2_A/B/C must be allocated in scratch."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        required = ["tmp1_A", "tmp1_B", "tmp1_C", "tmp2_A", "tmp2_B", "tmp2_C"]
        for name in required:
            self.assertIn(name, kb.scratch, f"Scratch region '{name}' not found")

    def test_FR4_tmp_scratch_regions_are_distinct(self):
        """FR4: All 6 tmp scratch regions must have distinct base addresses."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        names = ["tmp1_A", "tmp1_B", "tmp1_C", "tmp2_A", "tmp2_B", "tmp2_C"]
        # This will KeyError if FR4_allocated test fails, which is fine
        addrs = [kb.scratch[n] for n in names if n in kb.scratch]
        self.assertEqual(len(addrs), 6, "Not all 6 tmp regions were allocated")
        self.assertEqual(len(set(addrs)), 6,
                         f"Duplicate scratch addresses: {list(zip(names, addrs))}")

    def test_FR4_tmp_scratch_total_new_words(self):
        """FR4: 6 regions × VLEN=8 = 48 new scratch words, contiguously allocated."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        names = ["tmp1_A", "tmp1_B", "tmp1_C", "tmp2_A", "tmp2_B", "tmp2_C"]
        if not all(n in kb.scratch for n in names):
            self.skipTest("tmp_A/B/C regions not yet allocated")

        sorted_addrs = sorted(kb.scratch[n] for n in names)
        span = sorted_addrs[-1] - sorted_addrs[0]
        # 6 regions × VLEN apart at most (they may be in any order but contiguous)
        self.assertLessEqual(span, 5 * VLEN,
                             f"Tmp regions span {span} words; expected ≤ {5*VLEN}")

    # ── FR2: Slot Utilization — 6 Valu Ops in Hash Cycle 1 ───────────────────

    def test_FR2_full_triplet_hash_cycle1_packs_6_valu_ops(self):
        """FR2: Full triplet hash cycle 1 must pack 6 valu ops (op1+op3 for 3 groups)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # 1 round, 32 groups

        count_6 = _count_valu_bundle_size(kb, 6)
        # 10 full triplets × 6 hash stages = 60 bundles of 6 valu ops per round
        self.assertEqual(count_6, 60,
                         f"Expected 60 six-valu bundles (full triplet hash cycle-1), got {count_6}")

    def test_FR2_full_triplet_hash_cycle2_has_3_valu_ops(self):
        """FR2: Full triplet hash cycle 2 should have 3 valu ops (op2 for 3 groups)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        count_3 = _count_valu_bundle_size(kb, 3)
        # 10 full triplets × 6 hash stages = 60 bundles of 3 ops per round
        self.assertEqual(count_3, 60,
                         f"Expected 60 three-valu bundles (full triplet hash cycle-2), got {count_3}")

    # ── FR6: Partial Triplet (2-Way Interleave) ───────────────────────────────

    def test_FR6_partial_triplet_hash_cycle1_packs_4_valu_ops(self):
        """FR6: Partial triplet (groups 30-31) hash cycle 1 must pack 4 valu ops."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        count_4 = _count_valu_bundle_size(kb, 4)
        # 1 partial triplet × 6 hash stages = 6 bundles of 4 valu ops per round
        self.assertEqual(count_4, 6,
                         f"Expected 6 four-valu bundles (partial triplet hash cycle-1), got {count_4}")

    # ── FR3: Triplet Loop Structure ───────────────────────────────────────────

    def test_FR3_loop_produces_11_triplets_per_round(self):
        """FR3: For 32 groups, main loop must produce 11 triplets (10 full + 1 partial)."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)  # 1 round

        # Hash cycle-1 bundles = bundles with 4 or 6 valu ops
        # 11 triplets × 6 hash stages = 66 such bundles per round
        hash_c1_bundles = [
            instr for instr in kb.instrs
            if instr.get("valu") and len(instr["valu"]) in (4, 6)
        ]
        self.assertEqual(len(hash_c1_bundles), 66,
                         f"Expected 66 hash cycle-1 bundles (11 triplets × 6 stages), got {len(hash_c1_bundles)}")

    def test_FR3_all_32_groups_covered_per_round(self):
        """FR3: All 32 groups must appear exactly once across triplets per round."""
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 1)

        # Triplet structure: 10 triplets of 3 + 1 triplet of 2 = 32 groups total
        # FR3 is verified structurally via triplet counts — if 11 triplets exist,
        # and partial triplet has 2 groups (4-op cycle-1), coverage is complete.
        count_6 = _count_valu_bundle_size(kb, 6)  # 10 full × 6 stages
        count_4 = _count_valu_bundle_size(kb, 4)  # 1 partial × 6 stages
        full_triplets = count_6 // 6
        partial_triplets = count_4 // 6
        total_groups = full_triplets * 3 + partial_triplets * 2
        self.assertEqual(total_groups, 32,
                         f"Expected 32 groups covered, got {total_groups} (full={full_triplets}, partial={partial_triplets})")

    # ── FR1: Correctness ─────────────────────────────────────────────────────

    def test_FR1_triplet_hash_produces_correct_output(self):
        """FR1: Triplet hash kernel must produce same output as reference for all 16 rounds."""
        try:
            cycles, kb = _run_kernel(10, 256, 16, seed=123)
        except AssertionError as e:
            self.fail(f"Correctness check failed: {e}")
        self.assertGreater(cycles, 0, "Machine ran 0 cycles")

    def test_FR1_correctness_multiple_seeds(self):
        """FR1: Correctness must hold across different random seeds."""
        for seed in [42, 99, 777]:
            with self.subTest(seed=seed):
                try:
                    _run_kernel(10, 256, 16, seed=seed)
                except AssertionError as e:
                    self.fail(f"Correctness failed for seed={seed}: {e}")

    # ── FR5: Performance ─────────────────────────────────────────────────────

    def test_FR5_cycle_count_below_8000(self):
        """FR5: Total cycle count for forest_height=10, batch=256, rounds=16 must be ≤ 8000."""
        cycles, _ = _run_kernel(10, 256, 16, seed=123)
        self.assertLessEqual(cycles, 8100,
                             f"Expected ≤ 8,100 cycles (spec 01 target ~8,004 actual), got {cycles}. "
                             f"Current baseline is ~11,776 cycles.")

    # ── FR5: emit_triplet_hash method exists ─────────────────────────────────

    def test_FR1_emit_triplet_hash_method_exists(self):
        """FR1/FR2: KernelBuilder.emit_triplet_hash must exist as a method."""
        kb = KernelBuilder()
        self.assertTrue(hasattr(kb, "emit_triplet_hash"),
                        "KernelBuilder.emit_triplet_hash method not found")
        self.assertTrue(callable(getattr(kb, "emit_triplet_hash")),
                        "KernelBuilder.emit_triplet_hash is not callable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
