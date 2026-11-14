"""Build the full explanation document for one input file.

This module is both the library entry point (``explain_document``) and the
command entry point:

    python3 -m decision_explain_lab.explain --input <JSON> --output <JSON>
"""

import argparse
import math
import sys

from .attribution import attribution_report
from .context import (BASE_GROUPS, DERIVED_FEATURES, eligible_rows, normalise_reference,
                      normalise_row, population_summary, reference_for_case, reference_key,
                      support_report)
from .model import score_parts
from .recourse import recourse_report, validate_operations
from .version_bridge import version_bridge


METHOD_NOTES = [
    '归因对象是冻结模型的 logit，用原始字段分组的 Shapley 值分解；分段树贡献按实际评分路径计算，不折算成系数。',
    '分组的原始字段同时改动，派生特征由 transform 重新计算；派生特征不作为独立输入参与博弈。',
    '相关原始字段（尤其 network 与 app_context、calls 与 duration_s）的归因不唯一；输出同时给出换分组后的数值。',
    '把一组留作参考人群的值、其余组取当前请求，是干预式构造；它表示模型怎么读，不是业务操作的因果效果。',
    '树叶子是否在参考人群中出现过单独报告；参考人群未覆盖的请求会标记为外推。',
    '业务建议只使用输入声明的操作，重新经过原始变换与模型验证；年龄组与终端档位不可改变。',
    '版本比较在公共参考人群上重算两个模型，并同时给出两种顺序的分解与交互项。',
]


def explain_document(document):
    models = document['models']
    if not isinstance(models, dict) or not models:
        raise ValueError('models 必须是非空对象')
    for name, model in models.items():
        if not isinstance(model, dict) or 'coefficients' not in model or 'intercept' not in model:
            raise ValueError('模型 ' + name + ' 缺少 intercept 或 coefficients')
        if not isinstance(model['coefficients'], dict):
            raise ValueError('模型 ' + name + ' 的 coefficients 必须是对象')
        if not isinstance(model.get('trees', []), list):
            raise ValueError('模型 ' + name + ' 的 trees 必须是数组')
    cutoff = document.get('cutoff')
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or not math.isfinite(cutoff):
        raise ValueError('cutoff 必须是有限数值')
    target = document.get('target_probability')
    if not isinstance(target, (int, float)) or isinstance(target, bool) or not math.isfinite(target) \
            or not 0.0 <= target <= 1.0:
        raise ValueError('target_probability 必须是 0 到 1 之间的有限数值')
    maximum_cost = document.get('maximum_cost')
    if not isinstance(maximum_cost, (int, float)) or isinstance(maximum_cost, bool) \
            or not math.isfinite(maximum_cost) or maximum_cost < 0:
        raise ValueError('maximum_cost 必须是非负有限数值')
    if not isinstance(document.get('cases'), list) or not document['cases']:
        raise ValueError('cases 必须是非空数组')

    diagnostics = {
        'method_notes': METHOD_NOTES,
        'feature_contract': {
            'raw_fields': ['calls', 'duration_s', 'tier', 'network', 'app_available',
                           'selected_app', 'age_group', 'known_at'],
            'derived_features': [dict(item) for item in DERIVED_FEATURES],
            'mutual_constraints': [
                'app_available=false 时 selected_app 必须是 none，否则模型拒绝评分',
                'calls 不能为负，duration_s 必须为正',
            ],
            'future_outcome_reference': '参考人群中含有 known_at > cutoff 的行会被排除，不会作为解释基线。',
            'immutable_fields': list(document.get('protected', [])),
        },
        'grouping': [{'group': group['name'], 'raw_fields': list(group['raw_fields']),
                      'derived_features': list(group['derived_features'])}
                     for group in BASE_GROUPS],
        'warnings': [],
    }

    operations, exclusive_pairs, protected = validate_operations(document)
    diagnostic_protected = protected

    cases = []
    seen_case_ids = set()
    for index, raw_case in enumerate(document['cases']):
        case = normalise_row(raw_case, 'cases[' + str(index) + ']')
        if case['case_id'] is None:
            raise ValueError('cases[' + str(index) + '] 缺少 case_id')
        if case['case_id'] in seen_case_ids:
            raise ValueError('cases 含重复 case_id：' + case['case_id'])
        if case['known_at'] is not None and case['known_at'] > cutoff:
            raise ValueError('cases[' + str(index) + '] 的 known_at 晚于截断；当前请求的未来结局不能用于解释')
        seen_case_ids.add(case['case_id'])
        cases.append(case)

    # Reference populations, cut at the observation cutoff and with the
    # declared rows kept for reporting.  Future outcomes are never used as
    # background, and each request is additionally removed from its own
    # explanation population below.
    references = {}
    reference_reports = {}
    for name in sorted(models):
        declared = normalise_reference(document.get('reference_versions', {}).get(name),
                                       'reference_versions.' + name)
        eligible, excluded = eligible_rows(declared, cutoff, label='reference_versions.' + name)
        references[name] = {'declared': declared, 'used': eligible, 'excluded': excluded}
        reference_reports[name] = {'declared_size': len(declared), 'excluded': excluded}

    if document.get('reference') is not None:
        union = normalise_reference(document['reference'], 'reference')
        union_eligible, union_excluded = eligible_rows(union, cutoff, label='reference')
        union_ids = [row['case_id'] for row in union_eligible]
        used_ids = {row['case_id'] for name in references
                    for row in references[name]['used']}
        diagnostics['reference_union'] = {
            'declared_size': len(union),
            'eligible_size': len(union_eligible),
            'used_by_any_version': sorted(used_ids),
            'declared_only': sorted(set(union_ids) - used_ids),
            'excluded_by_cutoff_or_self': union_excluded,
            'note': '并集只用于核对；每版本解释使用各自的参考人群，固定人群比较另见 version_bridge。',
        }

    predictions = []
    attributions = []
    attribution_references = {}
    for case in cases:
        case_predictions = {'case_id': case['case_id'], 'models': {}}
        case_attributions = {'case_id': case['case_id'], 'models': {}}
        attribution_references[case['case_id']] = {}
        for name in sorted(models):
            reference = reference_for_case(references[name]['used'], case)
            if not reference:
                raise ValueError('参考人群在排除自引用后为空：' + name + ' / ' + case['case_id'])
            attribution_references[case['case_id']][name] = reference
            support = support_report(models[name], case, reference)
            parts = score_parts(models[name], case)
            case_predictions['models'][name] = {
                'logit': parts['logit'],
                'probability': parts['probability'],
                'linear_logit': parts['linear'],
                'tree_logit': sum(term['value'] for term in parts['tree_terms']),
                'tree_leaves_reached': [
                    {'tree_index': term['tree_index'], 'leaf_value': term['value']}
                    for term in parts['tree_terms']],
            }
            case_attributions['models'][name] = attribution_report(
                models[name], name, case, reference,
                dict(population_summary(reference),
                     declared_size=reference_reports[name]['declared_size'],
                     excluded=reference_reports[name]['excluded'],
                     excluded_case_ids=sorted(
                         row['case_id'] for row in references[name]['used']
                         if row['case_id'] not in {item['case_id'] for item in reference})),
                support)
        predictions.append(case_predictions)
        attributions.append(case_attributions)

    recourse = recourse_report(document, operations, exclusive_pairs, cases, models,
                               {name: references[name]['used'] for name in references},
                               diagnostics, protected)
    bridge = version_bridge(document, cases, attribution_references, diagnostics)

    diagnostics['protected_fields_checked'] = diagnostic_protected
    diagnostics['explanations_reliable'] = [
        {'case_id': item['case_id'], 'models': {
            name: {'reliable': item['models'][name]['support']['reliable_for_attribution'],
                   'support_status': item['models'][name]['support']['status'],
                   'reconstruction_exact': item['models'][name]['reconstruction']['exact'],
                   'max_abs_reconstruction_residual':
                       abs(item['models'][name]['reconstruction']['residual'])}
            for name in sorted(item['models'])}}
        for item in attributions
    ]
    diagnostics['recourse_summary'] = [
        {'case_id': item['case_id'], 'status': item['status'],
         'recommended': None if item['recommended_plan'] is None
         else item['recommended_plan']['operations'],
         'cost': None if item['recommended_plan'] is None
         else item['recommended_plan']['cost'],
         'extrapolates': None if item['recommended_plan'] is None
         else item['recommended_plan']['extrapolates_beyond_reference']}
        for item in recourse['cases']
    ]
    diagnostics['version_bridge_exact'] = all(
        case_report['exactness']['exact'] for case_report in bridge['cases'])
    diagnostics['attribution_exact'] = all(
        item['models'][name]['reconstruction']['exact']
        for item in attributions for name in item['models'])
    diagnostics['determinism'] = {
        'reference_fingerprints': {
            name: reference_key(references[name]['used']) for name in sorted(references)},
        'fixed_population_fingerprint': bridge['fixed_population']['fingerprint'],
        'note': '指纹只覆盖原始字段与 case_id，排序后计算；同输入重跑应得到相同结果。',
    }
    if any(not reliable['models'][name]['reliable']
           for reliable in diagnostics['explanations_reliable']
           for name in reliable['models']):
        diagnostics['warnings'].append(
            '至少一个请求或其解释落在参考人群覆盖之外；对应归因应标注为外推，不能当作稳定结论。')

    return {
        'schema_version': 1,
        'cutoff': cutoff,
        'predictions': predictions,
        'attributions': attributions,
        'recourse': recourse,
        'version_bridge': bridge,
        'diagnostics': diagnostics,
        'limitations': [
            '归因说明冻结模型如何使用输入，不说明现实中改变某个特征会改变结局。',
            '相关字段的归因不唯一；改变分组会改变各组的数值，虽然总和不变。',
            '业务操作建议是模型验证过的反事实，不是因果效果，也不包含未声明操作的收益。',
            '参考人群来自历史观测；未覆盖区域的结果属于外推。',
            '版本差异分解依赖公共参考人群；两版本人群差异过大时，交互项可能很大。',
            '未到达的结局没有被当作已观察的否定结果。',
        ],
    }


def main(argv=None):
    from .io import read_json, write_json

    parser = argparse.ArgumentParser(
        description='解释冻结的任务转人工模型：个体归因、可行改动建议与版本差异分析。')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        document = read_json(args.input)
        if not isinstance(document, dict):
            raise ValueError('输入顶层必须是对象')
        result = explain_document(document)
        write_json(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    print('结果已保存：' + args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
