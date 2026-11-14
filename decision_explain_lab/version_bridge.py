"""Version comparison that keeps reference-population change separate from
model change.

The two frozen models in this lab are explained against *different* declared
reference populations.  Subtracting one attribution vector from the other
would mix two different baselines and credit the model with a change caused by
swapping the comparison population.

This module recomputes both models on a shared fixed population — the rows
that the two declared populations have in common, matched by ``case_id`` and
by identical raw content — and reports two exact decompositions:

* primary: v1-native -> v1-fixed -> v2-fixed -> v2-native
* alternative: v1-native -> v2-at-v1-population -> v2-native

Each leg is a real recomputation.  Both chains sum to the same observed change
because they start and end at the same two native attributions.  The two
orderings differ in how much of the joint change they assign to the model
versus the population; that difference is reported as the interaction term
instead of being hidden.
"""

from .attribution import _coalition_values, shapley
from .context import (BASE_GROUPS, eligible_rows, normalise_reference, population_summary,
                      reference_key, reference_for_case)
from .model import RAW_FIELDS, score_parts


TOLERANCE = 1e-9


def _composition(rows):
    summary = population_summary(rows)
    return {'levels': summary['levels'], 'ranges': summary['ranges']}


def _row_content(row):
    """The raw model inputs of a reference row, ignoring provenance columns.

    Extra columns (for example source channel) may differ between the two
    snapshots without changing what the model sees.  Alignment therefore
    compares only the fields the model consumes plus the optional protection
    field, and keeps the first version's row content for scoring.
    """
    return tuple((field, row.get(field)) for field in RAW_FIELDS)


def _validate_version_rows(rows, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError(label + ' 必须是非空数组')
    seen = set()
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(label + '[' + str(index) + '] 必须是对象')
        case_id = row.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(label + '[' + str(index) + '] 缺少 case_id')
        if case_id in seen:
            raise ValueError(label + ' 含重复 case_id：' + case_id)
        seen.add(case_id)
        result.append(row)
    return result


def fixed_population(document, cutoff):
    """Shared population for the two-version recomputation."""
    versions = sorted(document['models'])
    reference_versions = document['reference_versions']
    missing = [name for name in versions if name not in reference_versions]
    if missing:
        raise ValueError('reference_versions 缺少模型版本：' + ', '.join(missing))
    extra = sorted(set(reference_versions) - set(versions))
    if extra:
        raise ValueError('reference_versions 含未定义的模型版本：' + ', '.join(extra))

    by_version = {name: _validate_version_rows(reference_versions[name],
                                               'reference_versions.' + name)
                  for name in versions}
    # Normalise before alignment so missing optional fields compare uniformly.
    normalised = {name: normalise_reference(rows, 'reference_versions.' + name)
                  for name, rows in by_version.items()}
    first, second = versions[0], versions[1]
    first_by_id = {row['case_id']: row for row in normalised[first]}
    second_by_id = {row['case_id']: row for row in normalised[second]}
    shared_ids = sorted(set(first_by_id) & set(second_by_id),
                        key=lambda value: (len(value), value))
    matched = []
    content_mismatch = []
    for case_id in shared_ids:
        if _row_content(first_by_id[case_id]) == _row_content(second_by_id[case_id]):
            matched.append(first_by_id[case_id])
        else:
            content_mismatch.append(case_id)
    if not matched:
        raise ValueError('两个版本的参考人群没有可对齐的公共记录，不能做固定人群比较')
    matched, excluded_by_cutoff = eligible_rows(matched, cutoff, label='公共参考人群')
    info = {
        'source': 'reference_versions 的公共记录',
        'method': 'align_by_case_id_and_identical_raw_model_inputs',
        'compared_fields': list(RAW_FIELDS),
        'note': '只比较模型实际读取的原始字段；额外的来源列不参与对齐，也不进入评分。',
        'size': len(matched),
        'fingerprint': reference_key(matched),
        'shared_case_ids': list(shared_ids),
        'first_version_size': len(by_version[first]),
        'second_version_size': len(by_version[second]),
        'first_only': len(set(first_by_id) - set(second_by_id)),
        'second_only': len(set(second_by_id) - set(first_by_id)),
        'content_mismatch_excluded': content_mismatch,
        'excluded_by_cutoff': excluded_by_cutoff,
    }
    return matched, info


def _cell(model, case, reference):
    values = _coalition_values(model, case, reference, BASE_GROUPS)
    parts = shapley(values, len(BASE_GROUPS))
    return {
        'baseline_logit': values[0],
        'groups': {group['name']: value for group, value in zip(BASE_GROUPS, parts)},
        'predicted_logit': values[(1 << len(BASE_GROUPS)) - 1],
    }


def _delta(left, right):
    return {group['name']: right['groups'][group['name']] - left['groups'][group['name']]
            for group in BASE_GROUPS}


def _sum(delta):
    return sum(delta.values())


def version_bridge(document, cases, attribution_references, diagnostics):
    versions = sorted(document['models'])
    if len(versions) != 2:
        raise ValueError('做版本差异分析需要恰好两个模型版本')
    first, second = versions
    fixed, fixed_info = fixed_population(document, document['cutoff'])
    result = {
        'versions': versions,
        'fixed_population': fixed_info,
        'method': {
            'primary_chain': [first + '_native', first + '_fixed',
                              second + '_fixed', second + '_native'],
            'alternative_chain': [first + '_native',
                                  second + '_at_' + first + '_population',
                                  second + '_native'],
            'legs': [
                'population_at_' + first + ': 固定 ' + first + '，参考人群从该版本原生人群换成公共人群',
                'model_at_fixed_population: 固定公共人群，把模型从 ' + first + ' 换成 ' + second,
                'population_at_' + second + ': 固定 ' + second + '，参考人群从公共人群换成该版本原生人群',
            ],
            'why_not_subtract': '两版本的原生参考人群不同，直接相减会把换人群造成的基线差异记到模型头上。',
            'interaction_note': '两种顺序都精确加总到同一变化；顺序之间的差额是模型效应与人群效应的交互项，任何可加分解都无法消除。',
        },
        'naive_difference_warning': (
            'naive_group_difference 保留为反例：它混合了人群变化与模型变化，不能解读为某个因素的模型贡献。'),
        'cases': [],
    }

    for case in cases:
        case_id = case['case_id']
        native_first = reference_for_case(attribution_references[case_id][first], case)
        native_second = reference_for_case(attribution_references[case_id][second], case)
        if not native_first or not native_second:
            raise ValueError('版本比较的参考人群在排除自引用后为空：' + case_id)
        fixed_for_case = reference_for_case(fixed, case)
        if not fixed_for_case:
            raise ValueError('固定人群在排除自引用后为空：' + case_id)

        cells = {
            first + '_native': _cell(document['models'][first], case, native_first),
            first + '_fixed': _cell(document['models'][first], case, fixed_for_case),
            second + '_fixed': _cell(document['models'][second], case, fixed_for_case),
            second + '_native': _cell(document['models'][second], case, native_second),
            second + '_at_' + first + '_population': _cell(
                document['models'][second], case, native_first),
        }
        an = cells[first + '_native']
        af = cells[first + '_fixed']
        bf = cells[second + '_fixed']
        bn = cells[second + '_native']
        b_at_first = cells[second + '_at_' + first + '_population']

        population_at_first = _delta(an, af)
        model_at_fixed = _delta(af, bf)
        population_at_second = _delta(bf, bn)
        model_at_first_population = _delta(an, b_at_first)
        population_at_second_alt = _delta(b_at_first, bn)
        interaction = {group['name']: model_at_fixed[group['name']] -
                       model_at_first_population[group['name']]
                       for group in BASE_GROUPS}
        naive = _delta(an, bn)
        primary_total = {group['name']: population_at_first[group['name']] +
                         model_at_fixed[group['name']] + population_at_second[group['name']]
                         for group in BASE_GROUPS}
        alternative_total = {group['name']: model_at_first_population[group['name']] +
                             population_at_second_alt[group['name']]
                             for group in BASE_GROUPS}

        # A cell's prediction = baseline + sum(group contributions).  A leg's
        # total effect therefore has two parts: the baseline shift and the sum
        # of the group deltas.  Both are reported; the leg is exact by
        # construction because it is a real recomputation of both endpoints.
        legs = [
            {
                'leg': 'population_at_' + first,
                'from': first + '_native',
                'to': first + '_fixed',
                'baseline_delta': af['baseline_logit'] - an['baseline_logit'],
                'group_deltas': population_at_first,
                'total_delta': af['predicted_logit'] - an['predicted_logit'],
            },
            {
                'leg': 'model_at_fixed_population',
                'from': first + '_fixed',
                'to': second + '_fixed',
                'baseline_delta': bf['baseline_logit'] - af['baseline_logit'],
                'group_deltas': model_at_fixed,
                'total_delta': bf['predicted_logit'] - af['predicted_logit'],
            },
            {
                'leg': 'population_at_' + second,
                'from': second + '_fixed',
                'to': second + '_native',
                'baseline_delta': bn['baseline_logit'] - bf['baseline_logit'],
                'group_deltas': population_at_second,
                'total_delta': bn['predicted_logit'] - bf['predicted_logit'],
            },
        ]
        alt_legs = [
            {
                'leg': 'model_at_' + first + '_population',
                'from': first + '_native',
                'to': second + '_at_' + first + '_population',
                'baseline_delta': b_at_first['baseline_logit'] - an['baseline_logit'],
                'group_deltas': model_at_first_population,
                'total_delta': b_at_first['predicted_logit'] - an['predicted_logit'],
            },
            {
                'leg': 'population_at_' + second,
                'from': second + '_at_' + first + '_population',
                'to': second + '_native',
                'baseline_delta': bn['baseline_logit'] - b_at_first['baseline_logit'],
                'group_deltas': population_at_second_alt,
                'total_delta': bn['predicted_logit'] - b_at_first['predicted_logit'],
            },
        ]

        v1_parts = score_parts(document['models'][first], case)
        v2_parts = score_parts(document['models'][second], case)
        expected = v2_parts['logit'] - v1_parts['logit']
        primary_sum = sum(leg['total_delta'] for leg in legs)
        alternative_sum = sum(leg['total_delta'] for leg in alt_legs)
        primary_residual = primary_sum - expected
        alternative_residual = alternative_sum - expected
        primary_group_sum = sum(_sum(leg['group_deltas']) + leg['baseline_delta'] for leg in legs)
        model_leg = legs[1]
        population_redistribution = sum(abs(value) for leg in legs
                                         if leg['leg'].startswith('population')
                                         for value in leg['group_deltas'].values())

        result['cases'].append({
            'case_id': case_id,
            'prediction_change': {
                'logit': expected,
                'probability': v2_parts['probability'] - v1_parts['probability'],
                'note': '预测只依赖请求本身，与参考人群无关；这一项是纯模型差异。',
            },
            'legs': legs,
            'alternative_order_legs': alt_legs,
            'group_deltas': {
                'naive_group_difference': naive,
                'population_at_' + first: population_at_first,
                'model_at_fixed_population': model_at_fixed,
                'population_at_' + second: population_at_second,
                'primary_chain_total': primary_total,
                'model_at_' + first + '_population': model_at_first_population,
                'population_at_' + second + '_under_' + first + '_ordering': population_at_second_alt,
                'alternative_chain_total': alternative_total,
                'interaction_model_minus_alternative': interaction,
            },
            'baseline_change': {
                'population_at_' + first: legs[0]['baseline_delta'],
                'model_at_fixed_population': legs[1]['baseline_delta'],
                'population_at_' + second: legs[2]['baseline_delta'],
                'total': bn['baseline_logit'] - an['baseline_logit'],
                'alternative_model_at_' + first + '_population': alt_legs[0]['baseline_delta'],
                'alternative_population_at_' + second: alt_legs[1]['baseline_delta'],
            },
            'totals': {
                'population_change_effect_on_prediction': sum(leg['total_delta'] for leg in legs
                                                              if leg['leg'].startswith('population')),
                'population_redistribution_logit': population_redistribution,
                'population_redistribution_note': (
                    '换参考人群不改变请求的预测，只会把解释数值在组间与基线之间挪动；'
                    '这一项是挪动幅度之和，不是预测变化。'),
                'model_change_effect_on_prediction': model_leg['total_delta'],
                'model_change_group_share': _sum(model_leg['group_deltas']),
                'model_change_baseline_share': model_leg['baseline_delta'],
                'interaction_logit': _sum(interaction),
                'model_change_at_' + first + '_population': alt_legs[0]['total_delta'],
            },
            'exactness': {
                'expected_total_logit_change': expected,
                'native_attribution_endpoint_change': bn['predicted_logit'] - an['predicted_logit'],
                'primary_chain_sum': primary_sum,
                'primary_chain_residual': primary_residual,
                'alternative_chain_sum': alternative_sum,
                'alternative_chain_residual': alternative_residual,
                'primary_chain_baseline_plus_groups': primary_group_sum,
                'exact': abs(primary_residual) <= TOLERANCE * max(1.0, abs(expected)) and
                         abs(alternative_residual) <= TOLERANCE * max(1.0, abs(expected)),
                'tolerance': TOLERANCE,
            },
            'cells': {
                name: {
                    'baseline_logit': value['baseline_logit'],
                    'predicted_logit': value['predicted_logit'],
                    'groups': value['groups'],
                }
                for name, value in cells.items()
            },
            'populations': {
                first + '_native': {'size': len(native_first),
                                    'fingerprint': reference_key(native_first),
                                    'composition': _composition(native_first)},
                second + '_native': {'size': len(native_second),
                                     'fingerprint': reference_key(native_second),
                                     'composition': _composition(native_second)},
                'fixed': {'size': len(fixed_for_case),
                          'fingerprint': reference_key(fixed_for_case),
                          'composition': _composition(fixed_for_case)},
            },
        })

    result['interpretation'] = {
        'population_change': '参考人群变化：同一模型、换参考人群造成的解释数值变化，不是模型行为改变。',
        'model_change': '模型变化：同一固定参考人群、换模型造成的解释数值变化。',
        'interaction': '交互项 = 公共人群上的模型变化 − 第一版人群上的模型变化；它衡量两种顺序对“模型贡献”的分歧。',
        'summary': _summary(result),
        'caveat': '即使固定人群，组间归因仍受分组方式影响；模型变化只说明冻结公式变化，不说明真实因果机制变化。',
    }
    diagnostics['version_bridge'] = {
        'fixed_population': {
            'size': fixed_info['size'],
            'fingerprint': fixed_info['fingerprint'],
            'first_only': fixed_info['first_only'],
            'second_only': fixed_info['second_only'],
            'content_mismatch_excluded': fixed_info['content_mismatch_excluded'],
        },
        'versions': versions,
        'exactness': {case_report['case_id']: case_report['exactness']
                      for case_report in result['cases']},
    }
    return result


def _summary(result):
    rows = []
    for case_report in result['cases']:
        totals = case_report['totals']
        population = totals['population_change_effect_on_prediction']
        model = totals['model_change_effect_on_prediction']
        rows.append({
            'case_id': case_report['case_id'],
            'total_logit_change': case_report['exactness']['expected_total_logit_change'],
            'population_change_effect_on_prediction': population,
            'model_change_effect_on_prediction': model,
            'interaction_logit': totals['interaction_logit'],
            'dominant': ('model' if abs(model) > abs(population) else 'population')
                        if (model or population) else 'none',
        })
    return rows
