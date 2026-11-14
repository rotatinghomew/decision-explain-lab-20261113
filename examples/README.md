# 端到端例子

三个输入都使用 `fixtures/explanations.json` 里的冻结模型，只改变参考人群、请求和操作集合。运行方式：

```bash
python3 -m decision_explain_lab.explain --input examples/<name>.json --output /tmp/<name>.json
```

| 输入 | 演示点 |
| --- | --- |
| `linked_operations.json` | 只有联动操作才可行：单步操作都达不到目标；v2 需要共享准备步骤 `prepare_wifi` 之后的“应用切换 + 联网切换”组合，`prepare_wifi` 成本只计一次。 |
| `outside_support.json` | 超出参考支持：`calls=40`、`duration_s=1` 远在参考范围之外，归因标记为外推；为它给出的建议也标记 `extrapolates_beyond_reference`。同一文件里另有一个支持范围内的请求作对照。 |
| `fixed_population_versions.json` | 固定人群重算两个版本：v1 与 v2 的参考人群只在公共记录上重叠，`version_bridge` 把人群变化与模型变化分开，并保留直接相减的“反例”字段。 |

输出中的 `diagnostics` 会记录每个例子的参考人群指纹（固定人群例子还包含公共人群大小），因此重跑可以与当时的事实核对。示例记录均为虚构，不对应真实设备或账户。
