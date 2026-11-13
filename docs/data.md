# 记录口径

加载冻结的任务转人工预测模型，执行特征变换、合法性检查和批量评分。

models 的 coefficients 对应 transform 返回的具名列，intercept 在 logit 空间；概率为 sigmoid(logit)。app_available 与 selected_app 有联动，call_density 从 calls、duration_s 派生。operations 是原始字段上的可执行动作，protected 是不可改变字段。

数据字段采用稳定的英文标识符。编号是业务身份，不由输入数组中的位置决定。未到达的结果与已经观察到的否定结果是不同事实。货币、时间和统计单位应按字段约定解释，不在读取时默默变换。

当前命令的输出只解释当前公开计算，不承诺示例覆盖全部业务情形。批量重放应使用新的输出位置，避免人工查看上一次运行留下的文件。
