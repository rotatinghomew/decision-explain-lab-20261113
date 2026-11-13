import math


def transform(row):
    if row['calls'] < 0 or row['duration_s'] <= 0:
        raise ValueError('调用次数或时长无效')
    if not row['app_available'] and row['selected_app'] != 'none':
        raise ValueError('选用应用与可用性矛盾')
    if row['selected_app'] not in {'travel', 'office', 'none'}:
        raise ValueError('应用类别未知')
    if row['tier'] not in {'entry', 'mid', 'high'}:
        raise ValueError('终端档位未知')
    if row['network'] not in {'weak', 'normal'}:
        raise ValueError('网络类别未知')
    density = row['calls'] / row['duration_s']
    return {'calls': float(row['calls']), 'call_density': density,
            'weak_network': float(row['network'] == 'weak'),
            'entry_device': float(row['tier'] == 'entry'),
            'travel_task': float(row['selected_app'] == 'travel'),
            'weak_travel': float(row['network'] == 'weak' and row['selected_app'] == 'travel')}


def score(model, row):
    features = transform(row)
    unknown = set(model['coefficients']) - set(features)
    if unknown:
        raise ValueError('模型含未知特征')
    logit = model['intercept'] + sum(features[key] * value for key, value in model['coefficients'].items())
    if not math.isfinite(logit):
        raise ValueError('模型分数非有限')
    probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    return {'logit': logit, 'probability': probability, 'features': features}


def inspect(document):
    return {'predictions': [{'case_id': row['case_id'], 'models':
             {name: score(model, row) for name, model in sorted(document['models'].items())}}
             for row in document['cases']]}
