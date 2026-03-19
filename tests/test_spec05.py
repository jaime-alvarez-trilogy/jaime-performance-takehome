"""
Spec 05 tests: ALU-assisted k=4 quadruplet hash pipeline.

Phase X.0: These tests define the expected behavior BEFORE implementation.
Tests that are currently RED (will become GREEN after implementation):
  - test_emit_alu_hash_stages_exists
  - test_emit_alu_hash_stages_mul_stage_2_cycles
  - test_emit_alu_hash_stages_xor_stage_3_cycles
  - test_emit_alu_hash_stages_total_15_cycles
  - test_alu_hash_matches_myhash
  - test_n_quadruplets_8

Tests that are currently GREEN (regression guards - must stay GREEN):
  - test_correctness_k4
  - test_no_cycle_regression
"""

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import unittest
from problem import HASH_STAGES, VLEN, myhash


def _run_kernel(forest_height=10, rounds=16, batch_size=256):
    """Helper: build and run kernel, return (machine, reference_mem)."""
    from frozen_problem import (
        Machine, build_mem_image, reference_kernel2,
        Tree, Input, N_CORES,
    )
    import random
    random.seed(42)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    from perf_takehome import KernelBuilder
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), batch_size, rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    return machine, ref_mem, kb


class TestFR3EmitAluHashStages(unittest.TestCase):
    """FR3: _emit_alu_hash_stages method exists and generates correct ALU ops."""

    def setUp(self):
        from perf_takehome import KernelBuilder
        from frozen_problem import Tree, Input, build_mem_image
        import random
        random.seed(0)
        self.KernelBuilder = KernelBuilder

    def _build_kb(self):
        from frozen_problem import Tree, Input, build_mem_image
        import random
        random.seed(0)
        forest_height = 10
        batch_size = 256
        rounds = 16
        kb = self.KernelBuilder()
        kb.build_kernel(forest_height, 2047, batch_size, rounds)
        return kb

    def test_emit_alu_hash_stages_exists(self):
        """FR3: KernelBuilder must have _emit_alu_hash_stages method."""
        self.assertTrue(
            hasattr(self.KernelBuilder, '_emit_alu_hash_stages'),
            "_emit_alu_hash_stages method must exist on KernelBuilder"
        )

    def test_emit_alu_hash_stages_mul_stage_2_cycles(self):
        """FR3: mul stage must emit exactly 2 cycles (16 ALU ops total)."""
        self.assertTrue(hasattr(self.KernelBuilder, '_emit_alu_hash_stages'),
                        "Requires _emit_alu_hash_stages to be implemented")
        kb = self._build_kb()
        a_D = kb.alloc_scratch("test_a_D", VLEN)
        tmp1_D = kb.alloc_scratch("test_tmp1_D", VLEN)
        tmp2_D = kb.alloc_scratch("test_tmp2_D", VLEN)
        alu_group_addr = (a_D, tmp1_D, tmp2_D)

        # Stage 0 is a mul stage: ("+", 0x7ED55D16, "+", "<<", 12)
        result = kb._emit_alu_hash_stages(alu_group_addr, stage_start=0, stage_end=1)
        self.assertEqual(len(result), 2, "Mul stage must emit 2 cycles")
        self.assertEqual(len(result[0]), VLEN, f"Mul cycle 1 must have {VLEN} ops (one per lane)")
        self.assertEqual(len(result[1]), VLEN, f"Mul cycle 2 must have {VLEN} ops (one per lane)")

    def test_emit_alu_hash_stages_xor_stage_3_cycles(self):
        """FR3: XOR stage must emit exactly 3 cycles (24 ALU ops total)."""
        self.assertTrue(hasattr(self.KernelBuilder, '_emit_alu_hash_stages'),
                        "Requires _emit_alu_hash_stages to be implemented")
        kb = self._build_kb()
        a_D = kb.alloc_scratch("test_a_D2", VLEN)
        tmp1_D = kb.alloc_scratch("test_tmp1_D2", VLEN)
        tmp2_D = kb.alloc_scratch("test_tmp2_D2", VLEN)
        alu_group_addr = (a_D, tmp1_D, tmp2_D)

        # Stage 1 is a XOR stage: ("^", 0xC761C23C, "^", ">>", 19)
        result = kb._emit_alu_hash_stages(alu_group_addr, stage_start=1, stage_end=2)
        self.assertEqual(len(result), 3, "XOR stage must emit 3 cycles")
        self.assertEqual(len(result[0]), VLEN, f"XOR cycle 1 must have {VLEN} ops")
        self.assertEqual(len(result[1]), VLEN, f"XOR cycle 2 must have {VLEN} ops")
        self.assertEqual(len(result[2]), VLEN, f"XOR cycle 3 must have {VLEN} ops")

    def test_emit_alu_hash_stages_total_15_cycles(self):
        """FR3: All 6 stages must emit exactly 15 cycles total."""
        self.assertTrue(hasattr(self.KernelBuilder, '_emit_alu_hash_stages'),
                        "Requires _emit_alu_hash_stages to be implemented")
        kb = self._build_kb()
        a_D = kb.alloc_scratch("test_a_D3", VLEN)
        tmp1_D = kb.alloc_scratch("test_tmp1_D3", VLEN)
        tmp2_D = kb.alloc_scratch("test_tmp2_D3", VLEN)
        alu_group_addr = (a_D, tmp1_D, tmp2_D)

        result = kb._emit_alu_hash_stages(alu_group_addr)
        # 3 mul stages × 2 cycles + 3 XOR stages × 3 cycles = 6 + 9 = 15
        self.assertEqual(len(result), 15, "All 6 stages must produce 15 cycles total")

    def test_alu_hash_matches_myhash(self):
        """FR3: ALU hash output must equal myhash() for known input values."""
        self.assertTrue(hasattr(self.KernelBuilder, '_emit_alu_hash_stages'),
                        "Requires _emit_alu_hash_stages to be implemented")
        from frozen_problem import Machine, N_CORES, SCRATCH_SIZE
        # Build a kernel, then manually simulate the ALU hash on a known input
        kb = self._build_kb()

        a_D = kb.alloc_scratch("test_a_match", VLEN)
        tmp1_D = kb.alloc_scratch("test_tmp1_match", VLEN)
        tmp2_D = kb.alloc_scratch("test_tmp2_match", VLEN)
        alu_group_addr = (a_D, tmp1_D, tmp2_D)
        alu_ops_by_cycle = kb._emit_alu_hash_stages(alu_group_addr)

        # Set initial values in scratch: a_D[0..7] = known input
        test_inputs = [0x12345678 + i for i in range(VLEN)]
        scratch = [0] * SCRATCH_SIZE
        for i in range(VLEN):
            scratch[a_D + i] = test_inputs[i]

        # Execute ALU ops cycle by cycle
        for cycle_ops in alu_ops_by_cycle:
            writes = {}
            for op, dest, src1, src2 in cycle_ops:
                s1 = scratch[src1]
                s2 = scratch[src2]
                if op == "*":
                    res = (s1 * s2) % (2**32)
                elif op == "+":
                    res = (s1 + s2) % (2**32)
                elif op == "^":
                    res = (s1 ^ s2) % (2**32)
                elif op == "<<":
                    res = (s1 << s2) % (2**32)
                elif op == ">>":
                    res = (s1 >> s2) % (2**32)
                else:
                    self.fail(f"Unknown op: {op}")
                writes[dest] = res
            # Commit writes (end-of-cycle)
            for dest, val in writes.items():
                scratch[dest] = val

        # Compare with reference myhash
        for i in range(VLEN):
            expected = myhash(test_inputs[i])
            actual = scratch[a_D + i]
            self.assertEqual(actual, expected,
                             f"Lane {i}: ALU hash {actual:#x} != myhash {expected:#x}")


class TestFR4QuadrupletLoop(unittest.TestCase):
    """FR4: Main loop must use 8 quadruplets per round (not 11 triplets)."""

    def test_n_quadruplets_8(self):
        """FR4: build_kernel must use n_quadruplets=8 instead of n_triplets=11."""
        from perf_takehome import KernelBuilder
        # Instrument: count xor bundles per round as proxy for triplet/quadruplet count.
        # A more direct check: look for n_quadruplets attribute or count addr patterns.
        # For now, check that nv_A is allocated for VLEN*4 (not VLEN*3).
        kb = KernelBuilder()
        kb.build_kernel(10, 2047, 256, 16)

        # After k=4, nv_A/nv_B must hold 4 groups (VLEN*4 = 32 words)
        # addr_next must hold 4 groups (VLEN*4 = 32 words)
        # We verify by checking scratch allocation gaps
        nv_a_addr = kb.scratch.get("nv_A")
        nv_b_addr = kb.scratch.get("nv_B")
        if nv_a_addr is not None and nv_b_addr is not None:
            # nv_B should be VLEN*4 = 32 words after nv_A
            self.assertEqual(nv_b_addr - nv_a_addr, VLEN * 4,
                             f"nv_A→nv_B gap should be {VLEN*4} (4 groups), got {nv_b_addr - nv_a_addr}")


class TestFR9FR10CorrectnessPerformance(unittest.TestCase):
    """FR9/FR10: Correctness and performance regression tests."""

    def test_correctness_k4(self):
        """FR9: Kernel must produce correct output matching reference."""
        machine, ref_mem, kb = _run_kernel()
        inp_values_p = ref_mem[6]

        from frozen_problem import VLEN as F_VLEN
        out_len = 256  # batch_size
        self.assertEqual(
            machine.mem[inp_values_p: inp_values_p + out_len],
            ref_mem[inp_values_p: inp_values_p + out_len],
            "Kernel output must match reference_kernel2"
        )

    def test_no_cycle_regression(self):
        """FR10: Cycle count must not regress above 2163."""
        machine, ref_mem, kb = _run_kernel()
        self.assertLessEqual(
            machine.cycle, 2163,
            f"Cycle count {machine.cycle} regressed above baseline 2163"
        )

    def test_cycle_improvement_target(self):
        """FR10 target: Cycle count should improve below 1790 (test_opus45_casual)."""
        machine, ref_mem, kb = _run_kernel()
        # This test is allowed to FAIL before full implementation but must PASS after.
        # It documents the target.
        self.assertLess(
            machine.cycle, 1790,
            f"Cycle count {machine.cycle} does not meet casual target of 1790"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
