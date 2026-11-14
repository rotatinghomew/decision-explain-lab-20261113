"""Input normalisation, feature grouping and reference-support checks.

The model consumes derived features that are not present in the request JSON.
Every intervention used for an explanation is therefore built from raw fields
and pushed back through :func:`decision_explain_lab.model.transform`.  A row
that the model would reject is never scored.
"""

import hashlib
import math

from .model import RAW_FIELDS, transform, tree_path


SCORING_FIELDS = ('calls', 'duration_s', 'tier', 'network', 'app_available', 'selected_app')
OPTIONAL_FIELDS = ('age_group', 'known_at')
NUMERIC_FIELDS = ('calls', 'duration_s')
CATEGORICAL_FIELDS = ('tier', 'network', 'app_available', 'selected_app', 'age_group')
# The joint-combination and conditional-range checks use only fields the model
# actually scores.  ``age_group`` is checked as an optional categorical field
# and never enters the model, so its absence is not a support failure.
JOINT_FIELDS = ('tier', 'network', 'app_available', 'selected_app')

# Grouping is part of the explanation contract.  A group is a set of raw
# fields that are changed together when a coalition is built.  Derived
# features are never treated as independent inputs: each one is owned by the
# raw group(s) that produce it.
BASE_GROUPS = (
    {'name': 'calls', 'raw_fields': ('calls',),
     'derived_features': ('calls', 'call_density')},
    {'name': 'duration_s', 'raw_fields': ('duration_s',),
     'derived_features': ('call_density',)},
    {'name': 'tier', 'raw_fields': ('tier',),
     'derived_features': ('entry_device',)},
    {'name': 'network', 'raw_fields': ('network',),
     'derived_features': ('weak_network', 'weak_travel')},
    {'name': 'app_context', 'raw_fields': ('app_available', 'selected_app'),
     'derived_features': ('travel_task', 'weak_travel')},
    {'name': 'age_group', 'raw_fields': ('age_group',),
     'derived_features': ()},
)

MERGE_TIME_GROUPS = (
    {'name': 'time_context', 'raw_fields': ('calls', 'duration_s'),
     'derived_features': ('calls', 'call_density')},
    {'name': 'tier', 'raw_fields': ('tier',), 'derived_features': ('entry_device',)},
    {'name': 'network', 'raw_fields': ('network',),
     'derived_features': ('weak_network', 'weak_travel')},
    {'name': 'app_context', 'raw_fields': ('app_available', 'selected_app'),
     'derived_features': ('travel_task', 'weak_travel')},
    {'name': 'age_group', 'raw_fields': ('age_group',), 'derived_features': ()},
)

MERGE_CONNECTIVITY_GROUPS = (
    {'name': 'calls', 'raw_fields': ('calls',),
     'derived_features': ('calls', 'call_density')},
    {'name': 'duration_s', 'raw_fields': ('duration_s',),
     'derived_features': ('call_density',)},
    {'name': 'tier', 'raw_fields': ('tier',), 'derived_features': ('entry_device',)},
    {'name': 'connectivity_app', 'raw_fields': ('network', 'app_available', 'selected_app'),
     'derived_features': ('weak_network', 'travel_task', 'weak_travel')},
    {'name': 'age_group', 'raw_fields': ('age_group',), 'derived_features': ()},
)

DERIVED_FEATURES = (
    {'name': 'calls', 'source_fields': ('calls',), 'formula': 'calls'},
    {'name': 'call_density', 'source_fields': ('calls', 'duration_s'),
     'formula': 'calls / duration_s'},
    {'name': 'weak_network', 'source_fields': ('network',),
     'formula': 'network == "weak"'},
    {'name': 'entry_device', 'source_fields': ('tier',),
     'formula': 'tier == "entry"'},
    {'name': 'travel_task', 'source_fields': ('selected_app',),
     'formula': 'selected_app == "travel"'},
    {'name': 'weak_travel', 'source_fields': ('network', 'selected_app'),
     'formula': 'network == "weak" and selected_app == "travel"'},
)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def normalise_row(row, label='记录'):
    _require(isinstance(row, dict), label + '必须是对象')
    for field in SCORING_FIELDS:
        _require(field in row, label + '缺少字段：' + field)
    calls = row['calls']
    duration = row['duration_s']
    _require(_is_number(calls), label + '的 calls 必须是有限数值')
    _require(_is_number(duration), label + '的 duration_s 必须是有限数值')
    _require(isinstance(row['app_available'], bool), label + '的 app_available 必须是布尔值')
    for field in ('tier', 'network', 'selected_app'):
        _require(isinstance(row[field], str), label + '的 ' + field + ' 必须是字符串')
    for field in ('age_group', 'known_at', 'case_id'):
        if row.get(field) is not None and field != 'known_at':
            _require(isinstance(row[field], str), label + '的 ' + field + ' 必须是字符串')
    if row.get('known_at') is not None:
        _require(_is_number(row['known_at']), label + '的 known_at 必须是有限数值')

    result = {field: row[field] for field in SCORING_FIELDS}
    result['age_group'] = row.get('age_group')
    result['known_at'] = row.get('known_at')
    result['case_id'] = row.get('case_id')
    # Validate the raw/derived contract before any explanation uses this row.
    transform(result)
    return result


def normalise_reference(rows, label):
    _require(isinstance(rows, list), label + '必须是数组')
    result = []
    seen = set()
    for index, row in enumerate(rows):
        item = normalise_row(row, label + '[' + str(index) + ']')
        _require(item['case_id'] is not None, label + '[' + str(index) + '] 缺少 case_id')
        _require(item['case_id'] not in seen, label + '含重复 case_id：' + item['case_id'])
        seen.add(item['case_id'])
        result.append(item)
    _require(bool(result), label + '不能为空')
    return result


def row_key(row):
    return tuple((field, row.get(field)) for field in RAW_FIELDS)


def reference_key(rows):
    """Order-independent fingerprint of a reference population."""
    digest = hashlib.sha256()
    for key in sorted(repr(row_key(row)) for row in rows):
        digest.update(key.encode('utf-8'))
        digest.update(b'\n')
    return 'ref-' + str(len(rows)) + '-' + digest.hexdigest()[:16]


def population_summary(rows):
    """Compact but explicit description of a reference population."""
    levels = {}
    for field in ('tier', 'network', 'selected_app', 'app_available'):
        counts = {}
        for row in rows:
            counts[str(row.get(field))] = counts.get(str(row.get(field)), 0) + 1
        levels[field] = counts
    calls = [row['calls'] for row in rows]
    durations = [row['duration_s'] for row in rows]
    return {
        'size': len(rows),
        'fingerprint': reference_key(rows),
        'levels': levels,
        'ranges': {
            'calls': [min(calls), max(calls)],
            'duration_s': [min(durations), max(durations)],
            'mean_call_density': sum(row['calls'] / row['duration_s'] for row in rows) / len(rows),
        },
    }


def reference_for_case(rows, case):
    """Reference rows usable for explaining ``case``.

    The explained request is removed from its own background population so no
    explanation is computed against itself.  ``known_at > cutoff`` rows were
    already removed by :func:`eligible_rows`.
    """
    case_id = case.get('case_id')
    return [row for row in rows if row.get('case_id') != case_id]


def eligible_rows(rows, cutoff, case_id=None, label='参考人群'):
    """Remove rows that were not known at the cutoff and the request itself.

    A row whose ``known_at`` is later than ``cutoff`` is a future outcome and
    must not be used as an explanation baseline.  The explained request is
    also removed from its own reference population so the explanation cannot
    lean on information from the same request.
    """
    eligible = []
    excluded = []
    for row in rows:
        known_at = row.get('known_at')
        if known_at is not None and known_at > cutoff:
            excluded.append({'case_id': row.get('case_id'), 'reason': 'known_after_cutoff',
                             'known_at': known_at})
            continue
        if case_id is not None and row.get('case_id') == case_id:
            excluded.append({'case_id': row.get('case_id'), 'reason': 'same_request_self_reference'})
            continue
        eligible.append(row)
    _require(bool(eligible), label + '在截断与自引用排除后为空')
    return eligible, excluded


def compose_row(case, background, groups, mask):
    """Build a hybrid row by substituting the selected raw groups."""
    row = {field: background[field] for field in RAW_FIELDS}
    for index, group in enumerate(groups):
        if mask & (1 << index):
            for field in group['raw_fields']:
                row[field] = case[field]
    row['case_id'] = case.get('case_id')
    return row


class SupportIndex:
    """Cheap support lookup for the reference population of one model.

    Support is deliberately descriptive: a row can be valid for the model and
    still lie outside the region the reference population actually covers.
    """

    def __init__(self, reference):
        self.reference = reference
        self.numeric_ranges = {}
        for field in NUMERIC_FIELDS:
            values = [row[field] for row in reference]
            self.numeric_ranges[field] = (min(values), max(values))
        self.category_values = {}
        for field in CATEGORICAL_FIELDS:
            self.category_values[field] = {row.get(field) for row in reference}
        self.joint_values = {
            tuple(row.get(field) for field in JOINT_FIELDS) for row in reference
        }
        self.conditional_ranges = {}
        for row in reference:
            key = tuple(row.get(field) for field in JOINT_FIELDS)
            low_calls, high_calls, low_duration, high_duration = self.conditional_ranges.get(
                key, (row['calls'], row['calls'], row['duration_s'], row['duration_s']))
            self.conditional_ranges[key] = (
                min(low_calls, row['calls']), max(high_calls, row['calls']),
                min(low_duration, row['duration_s']), max(high_duration, row['duration_s']))

    def classify_fields(self, row, fields):
        outside_fields = []
        unseen_categories = []
        for field in fields:
            if field in NUMERIC_FIELDS:
                low, high = self.numeric_ranges[field]
                if row[field] < low or row[field] > high:
                    outside_fields.append({'field': field, 'value': row[field],
                                           'reference_range': [low, high]})
            elif field in CATEGORICAL_FIELDS:
                # An optional field absent from the request is not compared.
                if row.get(field) is not None and row.get(field) not in self.category_values[field]:
                    unseen_categories.append({'field': field, 'value': row.get(field),
                                              'reference_values': sorted(
                                                  self.category_values[field],
                                                  key=lambda value: str(value))})
        return outside_fields, unseen_categories

    def classify(self, row):
        outside_fields, unseen_categories = self.classify_fields(row, RAW_FIELDS)
        joint_key = tuple(row.get(field) for field in JOINT_FIELDS)
        joint_seen = joint_key in self.joint_values
        conditional_outside = []
        if joint_seen:
            low_calls, high_calls, low_duration, high_duration = self.conditional_ranges[joint_key]
            if row['calls'] < low_calls or row['calls'] > high_calls:
                conditional_outside.append({'field': 'calls', 'value': row['calls'],
                                            'conditional_range': [low_calls, high_calls]})
            if row['duration_s'] < low_duration or row['duration_s'] > high_duration:
                conditional_outside.append({'field': 'duration_s', 'value': row['duration_s'],
                                            'conditional_range': [low_duration, high_duration]})
        if outside_fields or unseen_categories:
            status = 'outside_support'
        elif not joint_seen:
            status = 'joint_combination_unseen'
        elif conditional_outside:
            status = 'outside_conditional_support'
        else:
            status = 'supported'
        return {
            'status': status,
            'outside_fields': outside_fields,
            'unseen_categories': unseen_categories,
            'joint_combination_seen': joint_seen,
            'conditional_outside_fields': conditional_outside,
        }


def tree_support(model, row, reference):
    """Fraction of reference rows that reach the same leaf path."""
    features = transform(row)
    results = []
    for index, tree in enumerate(model.get('trees', [])):
        path, leaf = tree_path(tree, features)
        path_key = tuple((item['feature'], item['threshold'], item['branch']) for item in path)
        count = 0
        for background in reference:
            background_features = transform(background)
            background_path, _ = tree_path(tree, background_features)
            background_key = tuple((item['feature'], item['threshold'], item['branch'])
                                   for item in background_path)
            if background_key == path_key:
                count += 1
        fraction = count / len(reference)
        results.append({
            'tree_index': index,
            'path': [dict(item) for item in path],
            'leaf_value': leaf,
            'reference_matches': count,
            'reference_size': len(reference),
            'support_fraction': fraction,
            'supported': count > 0,
            'weak_support': 0 < fraction < 0.05,
        })
    return results


def support_report(model, row, reference):
    index = SupportIndex(reference)
    direct = index.classify(row)
    trees = tree_support(model, row, reference)
    tree_unsupported = any(not item['supported'] for item in trees)
    tree_weak = any(item['weak_support'] for item in trees)
    if direct['status'] == 'outside_support':
        status = 'outside_support'
    elif tree_unsupported:
        status = 'tree_path_unsupported'
    elif direct['status'] == 'joint_combination_unseen':
        status = 'joint_combination_unseen'
    elif direct['status'] == 'outside_conditional_support':
        status = 'outside_conditional_support'
    elif tree_weak:
        status = 'weak_support'
    else:
        status = 'supported'
    return {
        'status': status,
        'reference_size': len(reference),
        'outside_fields': direct['outside_fields'],
        'unseen_categories': direct['unseen_categories'],
        'joint_combination_seen': direct['joint_combination_seen'],
        'conditional_outside_fields': direct['conditional_outside_fields'],
        'tree_paths': trees,
        'reliable_for_attribution': status == 'supported',
        'notes': ([
            '该请求落在参考人群未覆盖的区域；归因仍是模型分解，但属于外推。'
        ] if status != 'supported' else []),
    }
