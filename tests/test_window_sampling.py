import unittest
from petlm.window_sampling import window_order, step_offsets, validate_schedule


class WindowSamplingTests(unittest.TestCase):
    def test_targets_never_overlap(self):
        order = window_order(49, 4, 7)
        positions = []
        for step in range(3):
            for micro in range(2):
                for offset in step_offsets(order, step, micro, 2, 2):
                    positions.extend(range(int(offset)+1, int(offset)+5))
        self.assertEqual(len(positions), 48)
        self.assertEqual(len(set(positions)), 48)

    def test_resume_uses_same_remaining_windows(self):
        a, b = window_order(65, 4, 42), window_order(65, 4, 42)
        self.assertEqual(step_offsets(a,2,1,2,2).tolist(),step_offsets(b,2,1,2,2).tolist())
        self.assertTrue(set(a[:8]).isdisjoint(b[8:]))

    def test_insufficient_corpus_is_rejected(self):
        config={'sampling':'nonoverlapping_single_pass','context':4,'seed':7,'steps':3,'batch_size':2,'accumulation':2}
        with self.assertRaises(ValueError):validate_schedule(48,config)
        self.assertEqual(len(validate_schedule(49,config)),12)
        with self.assertRaises(ValueError):validate_schedule(49,config,True)

    def test_no_wraparound_and_legacy_mode_preserved(self):
        with self.assertRaises(ValueError):step_offsets(window_order(9,4),1,0,2,1)
        self.assertIsNone(validate_schedule(9,{}))
        with self.assertRaises(ValueError):validate_schedule(9,{'sampling':'typo'})

    def test_validation_offsets_are_fixed_and_tail_is_dropped(self):
        self.assertEqual(window_order(12,4).tolist(),[0,4])
        with self.assertRaises(ValueError):window_order(4,4)


if __name__=='__main__':unittest.main()
