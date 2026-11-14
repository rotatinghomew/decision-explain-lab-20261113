"""Constrained counterfactual search over declared business operations.

An operation is a raw-field change: linked raw fields are applied together,
derived features are recomputed by :func:`transform`, and the resulting row
must pass the model's legality checks before it can be scored.  Costs are
charged once per distinct operation, so a preparation step shared by several
follow-up operations is not paid twice.

Nothing here is a causal effect.  The plan is a model-verified statement of
the form "if these declared actions are executed, the frozen model scores the
resulting request at ...".  Only operations supplied in the input can appear;
protected fields (in particular age group and device tier) can never change.
"""

import math

from .context import SCORING_FIELDS, SupportIndex
from .model import RAW_FIELDS, score_parts, transform


HARD_PROTECTED = ('age_group', 'tier')
OPERATION_FIELDS = ('calls', 'duration_s', 'tier', 'network', 'app_available', 'selected_app')
SEARCH_LIMIT = 200000


def _validate_change(field, value, operation_id):
    if field not in OPERATION_FIELDS:
        raise ValueError('操作 ' + operation_id + ' 试图改动不可用于业务操作或未知的字段：' + str(field))
    if field in HARD_PROTECTED:
        raise ValueError('操作 ' + operation_id + ' 试图改动保护字段：' + field)
    if field in ('calls', 'duration_s'):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('操作 ' + operation_id + ' 的 ' + field + ' 必须是有限数值')
        if value < 0:
            raise ValueError('操作 ' + operation_id + ' 会把 ' + field + ' 设为负值')
    elif field == 'app_available':
        if not isinstance(value, bool):
            raise ValueError('操作 ' + operation_id + ' 的 app_available 必须是布尔值')
    elif field == 'tier':
        if value not in {'entry', 'mid', 'high'}:
            raise ValueError('操作 ' + operation_id + ' 的终端档位未知')
    elif field == 'network':
        if value not in {'weak', 'normal'}:
            raise ValueError('操作 ' + operation_id + ' 的网络类别未知')
    elif field == 'selected_app':
        if value not in {'travel', 'office', 'none'}:
            raise ValueError('操作 ' + operation_id + ' 的应用类别未知')


def _validate_condition(field, value, operation_id):
    """A precondition only reads a field, so protected fields are allowed."""
    if field not in RAW_FIELDS:
        raise ValueError('操作 ' + operation_id + ' 的前置条件字段未知：' + str(field))
    if field in ('calls', 'duration_s'):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('操作 ' + operation_id + ' 的前置条件 ' + field + ' 必须是有限数值')
    elif field == 'app_available':
        if not isinstance(value, bool):
            raise ValueError('操作 ' + operation_id + ' 的前置条件 app_available 必须是布尔值')
    elif field == 'tier' and value not in {'entry', 'mid', 'high'}:
        raise ValueError('操作 ' + operation_id + ' 的前置条件终端档位未知')
    elif field == 'network' and value not in {'weak', 'normal'}:
        raise ValueError('操作 ' + operation_id + ' 的前置条件网络类别未知')
    elif field == 'selected_app' and value not in {'travel', 'office', 'none'}:
        raise ValueError('操作 ' + operation_id + ' 的前置条件应用类别未知')


def validate_operations(document):
    protected = document['protected']
    if not isinstance(protected, list) or not all(isinstance(field, str) for field in protected):
        raise ValueError('protected 必须是字段名数组')
    unknown_protected = set(protected) - set(RAW_FIELDS)
    if unknown_protected:
        raise ValueError('protected 含未知字段：' + ', '.join(sorted(unknown_protected)))
    # Device tier and age group are never actionable, whether or not the input
    # lists them.  This is the business contract, not a dataset preference.
    effective_protected = sorted(set(protected) | set(HARD_PROTECTED))

    operations = document['operations']
    if not isinstance(operations, list):
        raise ValueError('operations 必须是数组')
    by_id = {}
    order = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError('operations[' + str(index) + '] 必须是对象')
        identifier = operation.get('id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('operations[' + str(index) + '] 缺少合法 id')
        if identifier in by_id:
            raise ValueError('操作编号重复：' + identifier)
        cost = operation.get('cost')
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
            raise ValueError('操作 ' + identifier + ' 的 cost 必须是非负有限数值')
        changes = operation.get('changes')
        if not isinstance(changes, dict):
            raise ValueError('操作 ' + identifier + ' 的 changes 必须是对象')
        for field, value in changes.items():
            if field in effective_protected:
                raise ValueError('操作 ' + identifier + ' 试图改动保护字段：' + field)
            _validate_change(field, value, identifier)
        prerequisites = operation.get('prerequisites', [])
        if not isinstance(prerequisites, list) or not all(isinstance(item, str) for item in prerequisites):
            raise ValueError('操作 ' + identifier + ' 的 prerequisites 必须是操作编号数组')
        exclusive = operation.get('exclusive_with', [])
        if not isinstance(exclusive, list) or not all(isinstance(item, str) for item in exclusive):
            raise ValueError('操作 ' + identifier + ' 的 exclusive_with 必须是操作编号数组')
        preconditions = operation.get('preconditions', {})
        if not isinstance(preconditions, dict):
            raise ValueError('操作 ' + identifier + ' 的 preconditions 必须是对象')
        for field, value in preconditions.items():
            _validate_condition(field, value, identifier)
        by_id[identifier] = {
            'id': identifier,
            'changes': dict(changes),
            'cost': float(cost),
            'prerequisites': list(prerequisites),
            'exclusive_with': list(exclusive),
            'preconditions': dict(preconditions),
            'input_index': index,
        }
        order.append(identifier)

    for identifier, operation in by_id.items():
        for prerequisite in operation['prerequisites']:
            if prerequisite not in by_id:
                raise ValueError('操作 ' + identifier + ' 的前置依赖不存在：' + prerequisite)
        for other in operation['exclusive_with']:
            if other not in by_id:
                raise ValueError('操作 ' + identifier + ' 的互斥关系指向不存在的操作：' + other)
            if other == identifier:
                raise ValueError('操作 ' + identifier + ' 不能与自身互斥')
    _check_prerequisite_cycles(by_id)

    exclusive_pairs = set()
    for identifier, operation in by_id.items():
        for other in operation['exclusive_with']:
            pair = tuple(sorted((identifier, other)))
            exclusive_pairs.add(pair)
    return [by_id[identifier] for identifier in order], exclusive_pairs, effective_protected


def _check_prerequisite_cycles(by_id):
    state = {}

    def visit(identifier, trail):
        status = state.get(identifier)
        if status == 'done':
            return
        if status == 'active':
            raise ValueError('操作前置依赖成环：' + ' -> '.join(trail + [identifier]))
        state[identifier] = 'active'
        for prerequisite in by_id[identifier]['prerequisites']:
            visit(prerequisite, trail + [identifier])
        state[identifier] = 'done'

    for identifier in by_id:
        visit(identifier, [])


def _mutually_exclusive(identifier, applied, exclusive_pairs):
    for other in applied:
        if tuple(sorted((identifier, other))) in exclusive_pairs:
            return other
    return None


def _apply_changes(row, operation):
    candidate = dict(row)
    for field, value in operation['changes'].items():
        candidate[field] = value
    candidate['case_id'] = row.get('case_id')
    transform(candidate)  # raises on contradictions; caller decides what to do
    return candidate


def _can_apply(row, operation, applied, exclusive_pairs):
    if operation['id'] in applied:
        return False, 'already_applied'
    missing = [item for item in operation['prerequisites'] if item not in applied]
    if missing:
        return False, 'missing_prerequisite:' + ','.join(missing)
    for field, value in operation['preconditions'].items():
        if row.get(field) != value:
            return False, 'precondition_not_met:' + field
    conflict = _mutually_exclusive(operation['id'], applied, exclusive_pairs)
    if conflict is not None:
        return False, 'mutually_exclusive_with:' + conflict
    return True, 'ok'


def _state_signature(row):
    values = []
    for field in SCORING_FIELDS:
        value = row.get(field)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        values.append(value)
    return tuple(values)


def _sequence_key(plan):
    return (plan['cost'], len(plan['operations']), tuple(plan['operations']))


def enumerate_plans(operation_list, exclusive_pairs, case, maximum_cost):
    """Enumerate reachable plans under cost and dependency constraints."""
    plans = [{
        'operations': [],
        'cost': 0.0,
        'row': {field: case[field] for field in RAW_FIELDS},
        'dependency_edges': [],
        'shared_preparation': [],
    }]
    seen = {}
    explored = 0
    illegal_transitions = 0
    stack = [(dict(case), (), (), 0.0, ())]  # row, applied, sequence, cost, dependency edges
    while stack:
        row, applied, sequence, cost, edges = stack.pop()
        explored += 1
        if explored > SEARCH_LIMIT:
            raise ValueError('操作组合搜索超出安全上限，无法保证完备性')
        for operation in operation_list:
            if cost + operation['cost'] > maximum_cost + 1e-12:
                continue
            usable, _reason = _can_apply(row, operation, applied, exclusive_pairs)
            if not usable:
                continue
            try:
                candidate = _apply_changes(row, operation)
            except ValueError:
                illegal_transitions += 1
                continue
            new_cost = cost + operation['cost']
            new_applied = applied + (operation['id'],)
            new_sequence = sequence + (operation['id'],)
            new_edges = edges + tuple((item, operation['id']) for item in operation['prerequisites'])
            signature = (_state_signature(candidate), tuple(sorted(new_applied)))
            previous = seen.get(signature)
            if previous is not None and _sequence_key(previous) <= (new_cost, len(new_sequence), new_sequence):
                continue
            plan = {
                'operations': list(new_sequence),
                'cost': new_cost,
                'row': {field: candidate[field] for field in RAW_FIELDS},
                'dependency_edges': [list(edge) for edge in new_edges],
                'shared_preparation': [],
            }
            seen[signature] = plan
            stack.append((candidate, new_applied, new_sequence, new_cost, new_edges))
    plans.extend(seen.values())
    # Deduplicate by the resulting raw request, keeping the cheapest route.
    by_row = {}
    for plan in plans:
        key = _state_signature(plan['row'])
        incumbent = by_row.get(key)
        if incumbent is None or _sequence_key(plan) < _sequence_key(incumbent):
            by_row[key] = plan
    final_plans = sorted(by_row.values(), key=_sequence_key)
    used_operations = {identifier for plan in final_plans for identifier in plan['operations']}
    for plan in final_plans:
        plan['shared_preparation'] = _shared_preparation(plan, operation_list)
    return final_plans, {
        'states_explored': explored,
        'plans_after_dedup': len(final_plans),
        'illegal_transitions_skipped': illegal_transitions,
        'operations_used': sorted(used_operations),
        'cost_budget': maximum_cost,
    }


def _shared_preparation(plan, operation_list):
    by_id = {operation['id']: operation for operation in operation_list}
    applied = set(plan['operations'])
    shared = []
    for identifier in plan['operations']:
        dependents = [other for other in applied
                      if identifier in set(by_id[other]['prerequisites'])]
        if len(dependents) > 1:
            shared.append({'operation': identifier, 'used_by': sorted(dependents)})
    return shared


def _plan_cost_breakdown(plan, operation_list):
    by_id = {operation['id']: operation for operation in operation_list}
    return [{'operation': identifier, 'cost': by_id[identifier]['cost']}
            for identifier in plan['operations']]


def _plan_payload(plan, operation_list, models, support_indexes, target):
    breakdown = _plan_cost_breakdown(plan, operation_list)
    total_cost = sum(item['cost'] for item in breakdown)
    model_results = {}
    extrapolated = False
    target_met_all = True
    for name in sorted(models):
        parts = score_parts(models[name], plan['row'])
        support = support_indexes[name].classify(plan['row'])
        met = parts['probability'] <= target + 1e-12
        target_met_all = target_met_all and met
        if support['status'] != 'supported':
            extrapolated = True
        model_results[name] = {
            'logit': parts['logit'],
            'probability': parts['probability'],
            'target_met': met,
            'support_status': support['status'],
            'outside_fields': support['outside_fields'],
            'unseen_categories': support['unseen_categories'],
        }
    return {
        'operations': list(plan['operations']),
        'cost': plan['cost'],
        'cost_breakdown': breakdown,
        'cost_accounting': {
            'total_cost': total_cost,
            'charged_operations': [item['operation'] for item in breakdown],
            'note': '每个不同操作只计一次成本；共享准备步骤不会按使用次数重复计费。',
        },
        'shared_preparation': plan['shared_preparation'],
        'dependency_edges': plan['dependency_edges'],
        'final_request': {field: plan['row'][field] for field in SCORING_FIELDS},
        'final_request_with_identity_and_protected': {field: plan['row'][field] for field in RAW_FIELDS},
        'final_derived_features': transform(plan['row']),
        'unchanged_fields': {
            'tier': plan['row']['tier'],
            'age_group': plan['row']['age_group'],
            'note': '保护字段按原请求保留，任何操作都没有改动它们。',
        },
        'target_met_for_all_models': target_met_all,
        'extrapolates_beyond_reference': extrapolated,
        'models': model_results,
        'verified': {
            'raw_transform': 'passed',
            'recomputed_from_raw_values': True,
            'protected_fields_unchanged': True,
            'illegal_combinations': 'none',
        },
    }


def _payload_summary(payload):
    """Compact view used where the full payload is repeated."""
    return {
        'operations': payload['operations'],
        'cost': payload['cost'],
        'target_met_for_all_models': payload['target_met_for_all_models'],
        'models': {name: {'probability': item['probability'],
                          'target_met': item['target_met'],
                          'support_status': item['support_status']}
                   for name, item in payload['models'].items()},
        'extrapolates_beyond_reference': payload['extrapolates_beyond_reference'],
    }


def recourse_report(document, operations, exclusive_pairs, cases, models, references, diagnostics,
                    protected):
    from .context import reference_for_case

    target = document['target_probability']
    maximum_cost = document['maximum_cost']
    result = {
        'constraints': {
            'target_probability': target,
            'maximum_cost': maximum_cost,
            'protected_fields': list(protected),
            'operation_count': len(operations),
            'dependency_rules': [
                {'operation': operation['id'],
                 'cost': operation['cost'],
                 'prerequisites': operation['prerequisites'],
                 'preconditions': operation['preconditions'],
                 'exclusive_with': operation['exclusive_with'],
                 'changes': operation['changes']}
                for operation in operations
            ],
        },
        'cases': [],
    }
    for case in cases:
        plans, search = enumerate_plans(operations, exclusive_pairs, case, maximum_cost)
        support_indexes = {
            name: SupportIndex(reference_for_case(references[name], case))
            for name in models
        }
        payloads = [_plan_payload(plan, operations, models, support_indexes, target)
                    for plan in plans]
        baseline = {}
        baseline_met_all = True
        for name in sorted(models):
            parts = score_parts(models[name], case)
            met = parts['probability'] <= target + 1e-12
            baseline_met_all = baseline_met_all and met
            baseline[name] = {
                'logit': parts['logit'],
                'probability': parts['probability'],
                'target_met': met,
            }
        per_model = {}
        for name in sorted(models):
            target_plans = [payload for payload in payloads
                            if payload['models'][name]['target_met']]
            target_plans.sort(key=lambda payload: (
                payload['cost'], payload['models'][name]['probability'], payload['operations']))
            lowest = sorted(payloads, key=lambda payload: (
                payload['models'][name]['probability'], payload['cost'], payload['operations']))
            supported_target = [payload for payload in target_plans
                                if payload['models'][name]['support_status'] == 'supported']
            per_model[name] = {
                'baseline_probability': baseline[name]['probability'],
                'target_met_at_baseline': baseline[name]['target_met'],
                'cheapest_target_met_plan': target_plans[0] if target_plans else None,
                'supported_target_met_plan': supported_target[0] if supported_target else None,
                'lowest_probability_plan': lowest[0] if lowest else None,
            }
        joint = sorted(payloads, key=lambda payload: (
            payload['cost'],
            max(payload['models'][name]['probability'] for name in models),
            payload['operations']))
        feasible = [payload for payload in joint if payload['target_met_for_all_models']]
        if baseline_met_all:
            status = 'target_already_met'
            recommended = None
        elif feasible:
            status = 'feasible_for_both_versions'
            recommended = feasible[0]
        else:
            status = 'no_joint_plan_within_constraints'
            recommended = None
        best_effort = None
        if not feasible:
            best_effort = min(
                (payload for payload in joint if payload['operations']),
                key=lambda payload: (
                    max(payload['models'][name]['probability'] for name in models),
                    payload['cost'], payload['operations']),
                default=None,
            )
        disagreement = []
        if status == 'no_joint_plan_within_constraints':
            for name in sorted(models):
                plan = per_model[name]['cheapest_target_met_plan']
                if plan is not None:
                    failed = [other for other in sorted(models)
                              if not plan['models'][other]['target_met']]
                    disagreement.append({
                        'plan': plan['operations'],
                        'cost': plan['cost'],
                        'meets_target_for': [name],
                        'fails_target_for': failed,
                        'support_status': plan['models'][name]['support_status'],
                    })
        result['cases'].append({
            'case_id': case['case_id'],
            'baseline': baseline,
            'target_probability': target,
            'status': status,
            'status_meaning': _STATUS_MEANINGS[status],
            'recommended_plan': recommended,
            'best_effort_plan': best_effort,
            'version_specific_plans': {
                name: (_payload_summary(per_model[name]['cheapest_target_met_plan'])
                       if per_model[name]['cheapest_target_met_plan'] is not None else None)
                for name in sorted(per_model)
            },
            'model_disagreement': disagreement,
            'per_model': {
                name: {
                    'baseline_probability': item['baseline_probability'],
                    'target_met_at_baseline': item['target_met_at_baseline'],
                    'cheapest_target_met_plan': _payload_summary(item['cheapest_target_met_plan'])
                    if item['cheapest_target_met_plan'] is not None else None,
                    'supported_target_met_plan': _payload_summary(item['supported_target_met_plan'])
                    if item['supported_target_met_plan'] is not None else None,
                    'lowest_probability_plan': _payload_summary(item['lowest_probability_plan'])
                    if item['lowest_probability_plan'] is not None else None,
                }
                for name, item in per_model.items()
            },
            'search': search,
            'notes': _recourse_notes(status, recommended, best_effort, disagreement,
                                     maximum_cost, target),
        })
    diagnostics['recourse_searches'] = [
        {'case_id': case_report['case_id'], **case_report['search']}
        for case_report in result['cases']
    ]
    return result


_STATUS_MEANINGS = {
    'target_already_met': '当前请求在冻结模型下已低于目标概率。',
    'feasible_for_both_versions': '存在一份改动建议，在成本上限内同时让所有冻结模型低于目标概率。',
    'no_joint_plan_within_constraints': '成本上限内没有任何操作组合能同时让所有冻结模型低于目标概率；版本之间可能不一致。',
}


def _recourse_notes(status, recommended, best_effort, disagreement, maximum_cost, target):
    notes = []
    if status == 'target_already_met':
        notes.append('当前请求已低于目标概率，无需改动；不代表现实中已经发生转人工。')
    elif status == 'feasible_for_both_versions':
        if recommended['extrapolates_beyond_reference']:
            notes.append('建议方案达到目标，但结果落在参考人群覆盖范围之外；模型分数按冻结公式给出，属于外推。')
        shared = recommended['shared_preparation']
        if shared:
            for item in shared:
                notes.append('共享准备步骤 ' + item['operation'] + ' 被 ' +
                             '、'.join(item['used_by']) + ' 共同使用，成本只计一次。')
    else:
        notes.append('在给定成本上限 ' + _format_number(maximum_cost) +
                     ' 和目标概率 ' + _format_number(target) +
                     ' 下，没有任何操作组合能同时满足所有冻结模型；允许报告没有低成本可行建议。')
        if best_effort is not None:
            notes.append('最低概率的可行组合仍不达到目标，仅作为差距说明，不作为推荐。')
        if disagreement:
            notes.append('各模型对同一操作组合的判断不一致，不能只按一个版本推荐。')
    notes.append('建议是对声明操作做模型验证的反事实，不是因果效果，也不保证业务执行后结果必然如此。')
    return notes


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
