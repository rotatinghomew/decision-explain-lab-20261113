import copy
import json
import tempfile
import unittest
from pathlib import Path


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


from decision_explain_lab.model import transform, score, inspect


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.document = load('explanations.json')
        self.row = copy.deepcopy(self.document['cases'][0])

    def test_derived_feature(self):
        features = transform(self.row)
        self.assertEqual(features['call_density'], self.row['calls'] / self.row['duration_s'])

    def test_illegal_app(self):
        self.row['app_available'] = False
        self.row['selected_app'] = 'travel'
        with self.assertRaises(ValueError):
            transform(self.row)

    def test_probability_stable(self):
        self.assertEqual(score({'intercept': -1000, 'coefficients': {}}, self.row)['probability'], 0)
        self.assertEqual(score({'intercept': 1000, 'coefficients': {}}, self.row)['probability'], 1)

    def test_unknown_coefficient(self):
        with self.assertRaises(ValueError):
            score({'intercept': 0, 'coefficients': {'future_outcome': 1}}, self.row)

    def test_two_versions(self):
        result = inspect(self.document)
        self.assertEqual(len(result['predictions'][0]['models']), 2)

    def test_negative_calls(self):
        self.row['calls'] = -1
        with self.assertRaises(ValueError):
            transform(self.row)
