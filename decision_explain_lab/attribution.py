"""Model attribution for a single frozen model.

The explained quantity is the model's own logit.  It is decomposed with the
Shapley value over *raw-field groups* rather than over coefficients, so the
piecewise-constant tree contribution is accounted for exactly and the sum of
the parts reproduces the model score for every valid request.

Two caveats are part of the contract and are emitted in the output:

* The split between correlated raw fields is not unique.  Correlated
  ``app_context`` / ``network`` / time fields can be moved between groups by
  re-grouping; the package reports that sensitivity instead of pretending the
  per-group number is intrinsic to the model.
* Leaving a group at a reference value while the request's other groups are
  fixed is an interventional construction.  It is not the causal effect of a
  business action, because the reference population is observational and the
  model itself is not a causal graph.
"""

import math

from .context import (BASE_GROUPS, MERGE_CONNECTIVITY_GROUPS, MERGE_TIME_GROUPS,
                      compose_row)
from .model import score_parts


TOLERANCE = 1e-9


def _coalition_values(model, case, reference, groups):
    values = {}
    for mask in range(1 << len(groups)):
        total = 0.0
        for background in reference:
            hybrid = compose_row(case, background, groups, mask)
            # Any hybrid that the model rejects is a construction error, not
            # a zero contribution.  Raise instead of silently scoring it.
            total += score_parts(model, hybrid)['logit']
        values[mask] = total / len(reference)
    return values


def shapley(values, group_count):
    weights = [math.factorial(size) * math.factorial(group_count - size - 1) / math.factorial(group_count)
               for size in range(group_count)]
    parts = []
    for index in range(group_count):
        contribution = 0.0
        for mask in range(1 << group_count):
            if mask & (1 << index):
                continue
            size = bin(mask).count('1')
            contribution += weights[size] * (values[mask | (1 << index)] - values[mask])
        parts.append(contribution)
    return parts


def _baseline_detail(model, case, reference, groups):
    """Score the request and the mean pure-reference baseline."""
    case_parts = score_parts(model, case)
    baseline_total = 0.0
    for background in reference:
        baseline_total += score_parts(model, background)['logit']
    return case_parts, baseline_total / len(reference)


def _group_contributions_by_output(groups, shapley_parts):
    """Pair each group's Shapley value with the raw/derived fields it owns."""
    return [{'group': group['name'], 'raw_fields': list(group['raw_fields']),
             'derived_features': list(group['derived_features']),
             'logit_contribution': contribution}
            for group, contribution in zip(groups, shapley_parts)]


def attribution_report(model, model_name, case, reference, reference_info, support):
    """Full attribution detail for one (model, reference) pair."""
    groups = BASE_GROUPS
    values = _coalition_values(model, case, reference, groups)
    parts = shapley(values, len(groups))
    case_parts, baseline_mean = _baseline_detail(model, case, reference, groups)

    group_entries = _group_contributions_by_output(groups, parts)
    reconstructed = baseline_mean + sum(parts)
    residual = case_parts['logit'] - reconstructed

    # Coefficient table: linear terms only, kept identifiable from the tree.
    coefficient_rows = []
    for feature, coefficient in model['coefficients'].items():
        value = case_parts['features'][feature]
        coefficient_rows.append({
            'feature': feature,
            'coefficient': coefficient,
            'value': value,
            'linear_contribution': coefficient * value,
        })
    coefficient_rows.sort(key=lambda item: (-abs(item['linear_contribution']), item['feature']))

    tree_rows = []
    for term in case_parts['tree_terms']:
        tree_rows.append({
            'tree_index': term['tree_index'],
            'path': [dict(item) for item in term['path']],
            'leaf_value': term['value'],
        })

    # Sensitivity: re-grouping correlated fields changes the split even though
    # the total is fixed.  Report the spread per raw field.
    variants = []
    per_field = {}
    for name, variant_groups in (('default', groups),
                                 ('merge_time_context', MERGE_TIME_GROUPS),
                                 ('merge_connectivity_app', MERGE_CONNECTIVITY_GROUPS)):
        if name == 'default':
            variant_parts = parts
        else:
            variant_values = _coalition_values(model, case, reference, variant_groups)
            variant_parts = shapley(variant_values, len(variant_groups))
        if name != 'default':
            variants.append({
                'grouping': name,
                'groups': [{'group': group['name'],
                            'logit_contribution': contribution,
                            'raw_fields': list(group['raw_fields'])}
                           for group, contribution in zip(variant_groups, variant_parts)],
                'reconstructed_logit': variant_values[(1 << len(variant_groups)) - 1],
            })
        for group, contribution in zip(variant_groups, variant_parts):
            for field in group['raw_fields']:
                per_field.setdefault(field, []).append({
                    'grouping': name,
                    'group': group['name'],
                    'logit_contribution': contribution,
                })
    non_uniqueness = []
    for field in sorted(per_field):
        observations = per_field[field]
        field_values = [item['logit_contribution'] for item in observations]
        non_uniqueness.append({
            'raw_field': field,
            'observations': observations,
            'spread': max(field_values) - min(field_values),
        })

    # Stability: split the reference population deterministically and re-run
    # the same game on each part.
    stability = _stability(model, case, reference, values[(1 << len(groups)) - 1])

    exact = abs(residual) <= TOLERANCE * max(1.0, abs(case_parts['logit']))
    largest_group = max((abs(value) for value in parts), default=0.0)
    return {
        'model': model_name,
        'reference': reference_info,
        'reference_baseline_logit': baseline_mean,
        'reference_baseline_probability': _sigmoid(baseline_mean),
        'prediction': {
            'logit': case_parts['logit'],
            'probability': case_parts['probability'],
            'linear_logit': case_parts['linear'],
            'tree_logit': sum(term['value'] for term in case_parts['tree_terms']),
        },
        'groups': group_entries,
        'groups_total': sum(parts),
        'reconstruction': {
            'baseline_logit': baseline_mean,
            'sum_group_contributions': sum(parts),
            'reconstructed_logit': reconstructed,
            'model_logit': case_parts['logit'],
            'residual': residual,
            'exact': exact,
            'tolerance': TOLERANCE,
        },
        'grouping_contract': {
            'default_grouping': [group['name'] for group in groups],
            'unique': False,
            'note': '组间拆分依赖分组方式。相关字段合组或拆组会改变各组数值，但总和不随分组改变。',
            'largest_group_absolute_share': largest_group / max(1.0, abs(sum(parts))),
        },
        'linear_coefficient_view': {
            'note': '仅列线性项。树贡献不是系数，不能与系数表直接比较大小。',
            'rows': coefficient_rows,
            'intercept': model['intercept'],
        },
        'tree_contributions': tree_rows,
        'grouping_sensitivity': {
            'note': '同一模型、同一请求、同一参考人群，仅改变分组方式；总和不变，组间数值改变。',
            'variants': variants,
            'per_raw_field_non_uniqueness': non_uniqueness,
        },
        'stability': stability,
        'support': support,
    }


def _sigmoid(logit):
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1 + exponential)


def _stability(model, case, reference, model_logit):
    """Deterministic split of the reference population, re-running the game.

    The population is split into up to ``STRATA`` interleaved folds based on
    its size; each fold is a legitimate sub-population.  Report the spread of
    the group contributions so a reader can see how much of a number belongs
    to the model and how much to the chosen background.  Very small
    populations are split into halves if possible and otherwise reported as
    not computed -- never as zero variance.
    """
    size = len(reference)
    if size >= 8:
        strata = 4
        minimum_fold = 2
    elif size >= 4:
        strata = 2
        minimum_fold = 2
    else:
        return {'samples': 0, 'note': '参考人群太小，无法做分折稳定性重算；这里不报告零方差。',
                'group_std': None, 'max_abs_deviation': None}
    ordered = sorted(reference, key=lambda row: str(row['case_id']))
    samples = []
    for fold in range(strata):
        subset = [row for index, row in enumerate(ordered) if index % strata == fold]
        if len(subset) < minimum_fold:
            continue
        values = _coalition_values(model, case, subset, BASE_GROUPS)
        parts = shapley(values, len(BASE_GROUPS))
        samples.append(parts)
    if len(samples) < 2:
        return {'samples': len(samples), 'note': '参考人群不足以做稳定的分折重算。',
                'group_std': None, 'max_abs_deviation': None}
    means = [sum(column) / len(samples) for column in zip(*samples)]
    variances = [sum((value - means[index]) ** 2 for value in column) / (len(samples) - 1)
                 for index, column in enumerate(zip(*samples))]
    deviations = [max(abs(means[index] - value) for value in column)
                  for index, column in enumerate(zip(*samples))]
    largest = max(deviations)
    return {
        'samples': len(samples),
        'scheme': 'interleaved_folds_by_case_id',
        'folds': strata,
        'group_mean': [{'group': group['name'], 'logit_contribution': means[index]}
                       for index, group in enumerate(BASE_GROUPS)],
        'group_std': [{'group': group['name'], 'std': math.sqrt(variances[index])}
                      for index, group in enumerate(BASE_GROUPS)],
        'max_abs_deviation': largest,
        'relative_max_deviation': largest / max(1.0, abs(model_logit)),
        'note': '这是参考人群抽样敏感性，不是模型参数的不确定性。',
    }
