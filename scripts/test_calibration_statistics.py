import unittest
from summarize_calibration import metric, paired_interval


def item(hit, baseline, concepts=1):
    return {'concepts': ['x']*concepts, 'ranks': {
        'lens': {'6': [0 if hit else None]*concepts, '10': [None]*concepts},
        'logit-lens': {'6': [0 if baseline else None]*concepts, '10': [None]*concepts}}}


class ResearchStatistics(unittest.TestCase):
    def test_task_macro_does_not_reward_large_task_or_extra_concepts(self):
        groups = {'small': [item(True, False, 4)], 'large': [item(False, True)]*99}
        result = paired_interval(groups, 'lens', 'logit-lens', repeats=200)
        self.assertEqual(result['difference'], 0)
        self.assertEqual(result['paired_bootstrap_95_percent'], [0, 0])

    def test_paired_identical_methods_have_zero_uncertainty(self):
        groups = {'task': [item(True, True), item(False, False)]}
        result = paired_interval(groups, 'lens', 'logit-lens', repeats=200)
        self.assertEqual(result['paired_bootstrap_95_percent'], [0, 0])

    def test_uncovered_concepts_are_misses_and_layers_are_not_selected_afterward(self):
        example = item(True, False, 2)
        example['ranks']['lens']['6'] = [9, None]
        self.assertEqual(metric(example, 'lens', 10), .5)
        self.assertEqual(metric(example, 'lens', 5), 0)
        self.assertEqual(metric(example, 'lens', 10, 10), 0)
        self.assertAlmostEqual(metric(example, 'lens', 0), .05)


if __name__ == '__main__':
    unittest.main()
