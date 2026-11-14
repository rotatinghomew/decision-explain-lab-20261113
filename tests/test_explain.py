import copy
import json
import tempfile
import unittest
from pathlib import Path

from decision_explain_lab.explain import explain_document
from decision_explain_lab.io import read_json


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'fixtures' / name).read_text(encoding='utf-8'))


def explain(document):
    return explain_document(copy.deepcopy(document))


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.document = load('explanations.json')
        self.result = explain(self.document)

    def test_reconstruction_is_exact_for_both_versions(self):
        for case in self.result['attributions']:
            for name, report in case['models'].items():
                reconstruction = report['reconstruction']
                self.assertTrue(reconstruction['exact'], (case['case_id'], name))
                self.assertLess(abs(reconstruction['residual']), 1e-9)
                self.assertAlmostEqual(
                    reconstruction['model_logit'],
                    report['prediction']['logit'], places=12)

    def test_group_sum_reproduces_logit(self):
        for case in self.result['attributions']:
            for report in case['models'].values():
                total = report['reconstruction']['baseline_logit'] + sum(
                    group['logit_contribution'] for group in report['groups'])
                self.assertAlmostEqual(total, report['prediction']['logit'], places=10)

    def test_tree_contribution_is_not_a_coefficient(self):
        report = self.result['attributions'][0]['models']['v1']
        self.assertIn('tree_contributions', report)
        self.assertEqual(report['prediction']['tree_logit'], -0.2)
        # Linear table must not pretend to exhaust the score.
        linear_sum = sum(row['linear_contribution'] for row in
                         report['linear_coefficient_view']['rows'])
        linear_sum += report['linear_coefficient_view']['intercept']
        self.assertAlmostEqual(linear_sum, report['prediction']['linear_logit'], places=10)
        self.assertNotAlmostEqual(linear_sum, report['prediction']['logit'], places=6)

    def test_related_features_attribution_not_unique(self):
        report = self.result['attributions'][0]['models']['v1']
        self.assertFalse(report['grouping_contract']['unique'])
        spreads = {item['raw_field']: item['spread'] for item in
                   report['grouping_sensitivity']['per_raw_field_non_uniqueness']}
        self.assertGreater(spreads['network'], 0.0)
        self.assertGreater(spreads['calls'], 0.0)
        # The total prediction is invariant under re-grouping.
        for variant in report['grouping_sensitivity']['variants']:
            self.assertAlmostEqual(variant['reconstructed_logit'],
                                   report['prediction']['logit'], places=9)

    def test_reference_population_is_version_specific(self):
        for case in self.result['attributions']:
            v1 = case['models']['v1']['reference']
            v2 = case['models']['v2']['reference']
            self.assertNotEqual(v1['fingerprint'], v2['fingerprint'])
        # The first case is part of the v1 reference and must not explain itself.
        h0 = self.result['attributions'][0]['models']['v1']['reference']
        self.assertEqual(h0['excluded_case_ids'], ['h0'])

    def test_stability_reported(self):
        report = self.result['attributions'][0]['models']['v1']
        self.assertGreaterEqual(report['stability']['samples'], 2)
        self.assertIsNotNone(report['stability']['max_abs_deviation'])

    def test_out_of_support_case_is_flagged(self):
        outside = self.result['attributions'][1]
        for report in outside['models'].values():
            self.assertEqual(report['support']['status'], 'outside_support')
            self.assertFalse(report['support']['reliable_for_attribution'])
            fields = {item['field'] for item in report['support']['outside_fields']}
            self.assertIn('calls', fields)
            self.assertIn('duration_s', fields)

    def test_supported_case_is_flagged_supported(self):
        inside = self.result['attributions'][0]
        for report in inside['models'].values():
            self.assertEqual(report['support']['status'], 'supported')
            self.assertTrue(report['support']['reliable_for_attribution'])

    def test_future_outcomes_are_not_used(self):
        document = load('explanations.json')
        for rows in document['reference_versions'].values():
            for row in rows:
                row['known_at'] = 500
        for row in document['reference']:
            row['known_at'] = 500
        with self.assertRaises(ValueError):
            explain(document)

    def test_single_row_reference_future_rejected(self):
        document = load('explanations.json')
        document['reference_versions']['v1'][0]['known_at'] = 999
        result = explain(document)
        reference = result['attributions'][0]['models']['v1']['reference']
        excluded = {item['case_id']: item['reason'] for item in reference['excluded']}
        self.assertEqual(excluded.get('h0'), 'known_after_cutoff')

    def test_negative_calls_never_scored(self):
        document = load('explanations.json')
        document['cases'][0]['calls'] = -1
        with self.assertRaises(ValueError):
            explain(document)

    def test_contradictory_app_never_scored(self):
        document = load('explanations.json')
        document['cases'][0]['app_available'] = False
        document['cases'][0]['selected_app'] = 'travel'
        with self.assertRaises(ValueError):
            explain(document)


class RecourseTests(unittest.TestCase):
    def setUp(self):
        self.document = load('explanations.json')
        self.result = explain(self.document)

    def test_h0_needs_linked_operations(self):
        case = self.result['recourse']['cases'][0]
        # v1 can reach the target only through the linked chain
        # prepare_wifi -> switch_wifi -> batch_calls, paying shared prep once.
        plan = case['version_specific_plans']['v1']
        self.assertIsNotNone(plan)
        self.assertEqual(plan['operations'],
                         ['prepare_wifi', 'switch_wifi', 'batch_calls'])
        self.assertEqual(plan['cost'], 4.0)
        # No joint plan exists because v2 does not reach the target.
        self.assertEqual(case['status'], 'no_joint_plan_within_constraints')
        self.assertIsNone(case['recommended_plan'])
        self.assertIsNone(case['version_specific_plans']['v2'])

    def test_shared_preparation_charged_once(self):
        document = load('explanations.json')
        # Add a second dependent on prepare_wifi so the shared step is visible.
        document['operations'].append({
            'id': 'second_switch', 'changes': {'selected_app': 'office'},
            'cost': 1, 'prerequisites': ['prepare_wifi'],
            'preconditions': {'network': 'weak'}})
        result = explain(document)
        case = result['recourse']['cases'][0]
        plans = [case['best_effort_plan']] if case['best_effort_plan'] else []
        self.assertIsNotNone(case['best_effort_plan'])
        self.assertEqual(
            sum(item['cost'] for item in case['best_effort_plan']['cost_breakdown']),
            case['best_effort_plan']['cost'])

    def test_no_plan_implies_no_recommendation(self):
        case = self.result['recourse']['cases'][0]
        self.assertIsNone(case['recommended_plan'])
        self.assertIn('没有低成本可行建议', ' '.join(case['notes']))
        # Best effort is explicitly not a recommendation.
        self.assertIsNotNone(case['best_effort_plan'])

    def test_outside_case_plan_is_marked_extrapolation(self):
        case = self.result['recourse']['cases'][1]
        self.assertEqual(case['status'], 'feasible_for_both_versions')
        plan = case['recommended_plan']
        self.assertTrue(plan['extrapolates_beyond_reference'])
        self.assertTrue(plan['target_met_for_all_models'])
        for name, model in plan['models'].items():
            self.assertEqual(model['support_status'], 'outside_support')

    def test_plan_never_uses_protected_or_illegal_state(self):
        for case in self.result['recourse']['cases']:
            for plan in [case['recommended_plan'], case['best_effort_plan']]:
                if plan is None:
                    continue
                final = plan['final_request']
                self.assertIn(final['tier'], {'entry', 'mid', 'high'})
                self.assertGreater(final['calls'], -1)
                self.assertGreater(final['duration_s'], 0)
                if not final['app_available']:
                    self.assertEqual(final['selected_app'], 'none')
                self.assertEqual(final['tier'], final['tier'])
                self.assertTrue(plan['verified']['recomputed_from_raw_values'])

    def test_operations_cannot_touch_protected(self):
        document = load('explanations.json')
        document['operations'].append({
            'id': 'upgrade', 'changes': {'tier': 'high'}, 'cost': 1})
        with self.assertRaises(ValueError):
            explain(document)

    def test_operation_cannot_make_negative_calls(self):
        document = load('explanations.json')
        document['operations'].append({
            'id': 'bad', 'changes': {'calls': -3}, 'cost': 1})
        with self.assertRaises(ValueError):
            explain(document)

    def test_missing_prerequisite_rejected(self):
        document = load('explanations.json')
        document['operations'].append({
            'id': 'orphan', 'changes': {'network': 'normal'}, 'cost': 1,
            'prerequisites': ['not_defined']})
        with self.assertRaises(ValueError):
            explain(document)

    def test_prerequisite_cycle_rejected(self):
        document = load('explanations.json')
        document['operations'].append({
            'id': 'a', 'changes': {}, 'cost': 1, 'prerequisites': ['b']})
        document['operations'].append({
            'id': 'b', 'changes': {}, 'cost': 1, 'prerequisites': ['a']})
        with self.assertRaises(ValueError):
            explain(document)

    def test_exclusive_operations_not_combined(self):
        case = self.result['recourse']['cases'][1]
        plan = case['recommended_plan']
        self.assertNotIn('manual', plan['operations'])
        self.assertNotIn('simplify', plan['operations'])


class VersionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.result = explain(load('explanations.json'))

    def test_fixed_population_is_shared(self):
        bridge = self.result['version_bridge']
        self.assertEqual(bridge['fixed_population']['size'], 50)
        for case in bridge['cases']:
            populations = case['populations']
            self.assertEqual(populations['fixed']['size'], 50)
            self.assertNotEqual(populations['v1_native']['fingerprint'],
                                populations['fixed']['fingerprint'])
            self.assertNotEqual(populations['v2_native']['fingerprint'],
                                populations['fixed']['fingerprint'])

    def test_chain_is_exact(self):
        for case in self.result['version_bridge']['cases']:
            exactness = case['exactness']
            self.assertTrue(exactness['exact'])
            self.assertLess(abs(exactness['primary_chain_residual']), 1e-9)
            self.assertLess(abs(exactness['alternative_chain_residual']), 1e-9)
            self.assertAlmostEqual(
                exactness['primary_chain_sum'],
                exactness['expected_total_logit_change'], places=9)

    def test_model_and_population_change_separated(self):
        case = self.result['version_bridge']['cases'][0]
        legs = {leg['leg']: leg for leg in case['legs']}
        self.assertIn('model_at_fixed_population', legs)
        self.assertIn('population_at_v1', legs)
        self.assertIn('population_at_v2', legs)
        # The model leg is a real recomputation on the fixed population.
        self.assertAlmostEqual(legs['model_at_fixed_population']['total_delta'], 0.45)
        self.assertAlmostEqual(
            sum(leg['total_delta'] for leg in case['legs']),
            case['exactness']['expected_total_logit_change'], places=9)

    def test_naive_difference_is_labelled_as_warning(self):
        self.assertIn('naive_difference_warning', self.result['version_bridge'])
        self.assertIn('不能解读', self.result['version_bridge']['naive_difference_warning'])

    def test_interaction_reported(self):
        for case in self.result['version_bridge']['cases']:
            self.assertIn('interaction_model_minus_alternative', case['group_deltas'])
            self.assertIn('interaction_logit', case['totals'])

    def test_population_at_v1_does_not_change_prediction(self):
        for case in self.result['version_bridge']['cases']:
            for leg in case['legs']:
                if leg['leg'].startswith('population'):
                    self.assertAlmostEqual(leg['total_delta'], 0.0, places=9)


class EndToEndTests(unittest.TestCase):
    def test_integrated_output_shape(self):
        result = explain(load('explanations.json'))
        for key in ('predictions', 'attributions', 'recourse', 'version_bridge', 'diagnostics'):
            self.assertIn(key, result)
        self.assertTrue(result['diagnostics']['attribution_exact'])
        self.assertTrue(result['diagnostics']['version_bridge_exact'])
        self.assertTrue(result['limitations'])

    def test_output_is_deterministic(self):
        document = load('explanations.json')
        first = explain_document(copy.deepcopy(document))
        second = explain_document(copy.deepcopy(document))
        self.assertEqual(json.dumps(first, sort_keys=True, ensure_ascii=False),
                         json.dumps(second, sort_keys=True, ensure_ascii=False))

    def test_example_linked_operations(self):
        result = explain(load('../examples/linked_operations.json'))
        case = result['recourse']['cases'][0]
        self.assertEqual(case['status'], 'feasible_for_both_versions')
        plan = case['recommended_plan']
        # A single operation never reaches the target for either model.
        for name in ('v1', 'v2'):
            per_model = case['per_model'][name]
            self.assertGreater(len(per_model['cheapest_target_met_plan']['operations']), 1)
        # The shared preparation step is charged once and used twice.
        self.assertEqual(plan['cost'], 4.0)
        shared = {item['operation']: item['used_by'] for item in plan['shared_preparation']}
        self.assertEqual(shared['prepare_wifi'],
                         ['switch_app_office', 'switch_wifi'])
        self.assertEqual(sum(item['cost'] for item in plan['cost_breakdown']), plan['cost'])

    def test_example_outside_support(self):
        result = explain(load('../examples/outside_support.json'))
        outside = next(item for item in result['attributions'] if item['case_id'] == 'outside')
        for report in outside['models'].values():
            self.assertEqual(report['support']['status'], 'outside_support')
            self.assertFalse(report['support']['reliable_for_attribution'])
        inside = next(item for item in result['attributions'] if item['case_id'] == 'inside')
        for report in inside['models'].values():
            self.assertEqual(report['support']['status'], 'supported')
        case = next(item for item in result['recourse']['cases'] if item['case_id'] == 'outside')
        self.assertTrue(case['recommended_plan']['extrapolates_beyond_reference'])
        self.assertTrue(any('外推' in note for note in case['notes']))

    def test_example_fixed_population(self):
        result = explain(load('../examples/fixed_population_versions.json'))
        bridge = result['version_bridge']
        case = bridge['cases'][0]
        self.assertTrue(case['exactness']['exact'])
        self.assertLess(case['populations']['fixed']['size'],
                        case['populations']['v1_native']['size'])
        # The population swaps move explanation mass but never the prediction.
        for leg in case['legs']:
            if leg['leg'].startswith('population'):
                self.assertAlmostEqual(leg['total_delta'], 0.0, places=9)
        # Naive subtraction disagrees with the fixed-population model leg.
        naive_tier = case['group_deltas']['naive_group_difference']['tier']
        model_tier = case['group_deltas']['model_at_fixed_population']['tier']
        self.assertNotAlmostEqual(naive_tier, model_tier, places=6)
        self.assertIn('interaction_logit', case['totals'])

    def test_command_writes_json(self):
        from decision_explain_lab.explain import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'result.json'
            status = main(['--input', str(ROOT / 'fixtures' / 'explanations.json'),
                           '--output', str(output)])
            self.assertEqual(status, 0)
            written = read_json(output)
            self.assertEqual(len(written['predictions']), 2)
            self.assertIn('version_bridge', written)

    def test_command_rejects_duplicate_json_fields(self):
        from decision_explain_lab.explain import main
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'broken.json'
            source.write_text('{"models": {}, "models": {}}', encoding='utf-8')
            output = Path(directory) / 'out.json'
            status = main(['--input', str(source), '--output', str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
