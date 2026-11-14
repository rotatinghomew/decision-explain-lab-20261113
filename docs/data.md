# 记录口径

冻结模型支持线性项与 `trees` 中的分段树贡献。树的内部节点按具名特征的 `threshold` 左闭比较选择 `left` 或 `right`，叶节点的 `value` 加入 logit。`operations` 的前置条件针对原始字段，`prerequisites` 和 `exclusive_with` 使用操作编号。`reference_versions` 分别保存两个版本当时使用的参考人群。

加载冻结的任务转人工预测模型，执行特征变换、合法性检查和批量评分。

models 的 coefficients 对应 transform 返回的具名列，intercept 在 logit 空间；概率为 sigmoid(logit)。app_available 与 selected_app 有联动，call_density 从 calls、duration_s 派生。operations 是原始字段上的可执行动作，protected 是不可改变字段。

数据字段采用稳定的英文标识符。编号是业务身份，不由输入数组中的位置决定。未到达的结果与已经观察到的否定结果是不同事实。货币、时间和统计单位应按字段约定解释，不在读取时默默变换。

当前命令的输出只解释当前公开计算，不承诺示例覆盖全部业务情形。批量重放应使用新的输出位置，避免人工查看上一次运行留下的文件。

## 解释口径

- 归因对象是模型 logit，按原始字段分组做 Shapley 分解。`calls`/`duration_s` 与 `network`/`app_available`/`selected_app` 的拆分由分组契约决定，不是数据内在属性；换分组会改变组间数值但不改变总和。派生特征不单独参赛，随其原始字段联动重算。
- 分段树贡献按实际路径读取叶值；`linear_coefficient_view` 只列线性项，不能与树贡献直接比较或相加后当作完整解释。
- `cutoff` 是观测截断：`known_at > cutoff` 的参考记录视为尚未观察的结局，不参与基线或比较；被解释请求也从自己的参考人群中排除。
- 参考支持分四层：数值范围、类别取值、模型评分字段的联合组合、树路径覆盖率。任一不满足即标记外推。
- `operations` 的 `changes` 可联动多个原始字段；派生特征由变换重算，非法组合（负数调用、不可用却选用应用）在评分前被拒绝。`protected` 之外，`tier` 与 `age_group` 永远不可操作。成本按不同操作计一次，共享准备步骤不重复计费。
- 版本比较必须先把两版 `reference_versions` 对齐到公共记录（按 `case_id` 与模型实际读取的原始字段比较，额外来源列不参与），再分别重算。两版原生人群不同时，直接相减的解释差不是模型贡献。

