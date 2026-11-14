# decision_explain_lab

加载冻结的任务转人工预测模型，执行特征变换、合法性检查和批量评分；在冻结模型上提供个体解释、受约束的可行改动建议与版本差异分析，不重新训练模型。

当前命令是 `explain`。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q decision_explain_lab
python3 -m unittest discover -s tests -v

python3 -m decision_explain_lab.explain --input fixtures/explanations.json --output /tmp/decision-explanations.json

# 三个端到端例子
python3 -m decision_explain_lab.explain --input examples/linked_operations.json --output /tmp/linked.json
python3 -m decision_explain_lab.explain --input examples/outside_support.json --output /tmp/outside.json
python3 -m decision_explain_lab.explain --input examples/fixed_population_versions.json --output /tmp/fixed.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

`python3 -m decision_explain_lab inspect --input <JSON> --output <JSON>` 仍然只做评分核对。

## 输出结构

`explain` 的结果是一个对象，包含：

| 字段 | 内容 |
| --- | --- |
| `predictions` | 每个请求、每个版本的 logit、概率、线性项与树项总和，以及实际到达的树叶。 |
| `attributions` | 每个请求、每个版本的模型归因：参考人群、基线、原始字段分组贡献、还原误差、分组敏感性、稳定性、支持范围。 |
| `recourse` | 受 `cost`、`prerequisites`/`preconditions`、`exclusive_with` 与 `protected` 约束的可行改动建议，重新经过 `transform` 与模型验证；允许没有建议。 |
| `version_bridge` | 在公共参考人群上重算两个版本，把“人群变化”和“模型变化”分开，并给出两种顺序的分解与交互项。 |
| `diagnostics` | 方法说明、特征契约、参考人群指纹、搜索规模、外推警告等。 |
| `limitations` | 本次输出不适用的结论边界。 |

## 归因

- 归因对象是冻结模型的 logit；用**原始字段分组**的 Shapley 值分解。派生特征不作为独立输入，由分组内的原始字段重新计算。
- 分段树的贡献按实际评分路径计入，不折算成系数，因此 `linear_coefficient_view` 只覆盖线性项。
- 还原误差显式给出：`baseline + 各组贡献` 与实际 logit 的残差；默认分组下应精确复原。
- **相关特征的归因不唯一**：`network` 与 `app_context`、`calls` 与 `duration_s` 拆组或合组会改变组间数值，总和不随分组改变。输出同时报告多种分组的数值与每个原始字段的跨度。
- 稳定性用参考人群的确定性分折重算；它衡量背景抽样敏感性，不是模型参数不确定性。
- 支持范围分别检查原始字段数值范围、类别、联合组合、条件范围与树路径覆盖率。超出参考人群的请求标记为外推，`reliable_for_attribution` 为 false。

## 可行改动建议

- 只使用输入声明的操作；一个操作可以联动多个原始字段（例如同时改 `app_available` 与 `selected_app`），派生特征与合法组合由 `transform` 重新计算。
- 成本按**不同操作**计一次；共享准备步骤只计一次，并明确列出被哪些操作共同使用。
- `prerequisites` 决定先后顺序，`preconditions` 检查当时字段，`exclusive_with` 禁止同时出现，成环或引用不存在操作会直接报错。
- `protected` 与不可变的 `tier`、`age_group` 永远不会被改动，建议里明确报告其原值。
- 到达目标概率时，同时验证所有冻结版本；若两个版本不一致，报告分歧而不是按某个版本硬推荐。没有可行组合时返回 null，并在 `notes` 中说明可以报告“没有低成本可行建议”。
- 建议是对声明操作做模型验证的反事实，**不是因果效果**。

## 版本比较

- 两个版本的原生参考人群不同，直接相减会把“换人群”记成“模型变化”。
- `version_bridge` 先在两个版本共同的记录上建立固定人群，再分别重算：

  `v1_native → v1_fixed → v2_fixed → v2_native`

  每条腿的 `baseline_delta + 各组分变化` 等于该腿的 `total_delta`，整条链精确等于请求本身的模型分差。
- 换参考人群不改变请求的预测，只改变解释数值的分配；`totals.population_change_effect_on_prediction` 因此接近 0，另有 `population_redistribution_logit` 描述挪动幅度。
- 同时也给出以第一版人群为起点的另一种顺序，两者之差作为交互项，不隐藏。
- `naive_group_difference` 只作为反例保留，不能解读为某个因素的模型贡献。

## 数据与目录

`decision_explain_lab/model.py` 提供冻结模型的评分与树路径；`context.py` 负责输入校验、派生特征契约、参考人群与支持范围；`attribution.py` 做分组 Shapley；`recourse.py` 做受限操作搜索；`version_bridge.py` 做版本分解；`explain.py` 组装输出并提供命令行。`fixtures/` 是可重放的演练记录；`examples/` 是三个端到端输入；`tests/` 验证领域边界与输入输出行为。

models 的 coefficients 对应 transform 返回的具名列，intercept 在 logit 空间；概率为 sigmoid(logit)。app_available 与 selected_app 有联动，call_density 从 calls、duration_s 派生。operations 是原始字段上的可执行动作，protected 是不可改变字段。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
