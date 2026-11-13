# decision_explain_lab

加载冻结的任务转人工预测模型，执行特征变换、合法性检查和批量评分。

当前公开命令是 `inspect`，用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q decision_explain_lab
python3 -m unittest discover -s tests -v
python3 -m decision_explain_lab inspect --input fixtures/explanations.json --output /tmp/decision-scores.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`decision_explain_lab/model.py` 提供当前观测计算；`decision_explain_lab/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

models 的 coefficients 对应 transform 返回的具名列，intercept 在 logit 空间；概率为 sigmoid(logit)。app_available 与 selected_app 有联动，call_density 从 calls、duration_s 派生。operations 是原始字段上的可执行动作，protected 是不可改变字段。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
