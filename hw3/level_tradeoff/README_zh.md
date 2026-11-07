# HW3：整体水平使用强度与增益稳健性

本目录实现用户确认的探索性后续实验：固定 HW2 的相对输入与位置拼接读出，增加两个整体幅度相关特征的分支，并比较训练时水平分支丢弃概率 p=0、0.25、0.5、0.75。每个条件使用种子 7、11、23。

先读 [实验前方案](protocol-zh.txt)。结果与方案分别保存，不能把观察后的解释写成事前预测。

## 文件

- `experiment.py`：双分支模型、训练、验证选择、测试和固定表示的水平干预。
- `baseline_model.py`、`features.py`：复制自本机 HW2 的模型与特征代码，来源为 [HW2](https://github.com/georg11a/deep-learning-assignment2)。
- `summarize.py`：聚合三种子结果和绘制验证、测试曲线。
- `results/config.json`：数据、代码、方案哈希和演员划分；`selection.json` 保存测试评分前冻结的验证选择。
- `results/val-summary.csv`、`test-summary.csv`：总体准确率、macro-F1、六类召回率和增益翻转率。类别数值不是 precision。
- `results/*-predictions.npz`：逐条类别概率和对应索引；`gain-features.npz` 是重提取特征。

## 运行

使用与 HW2 相同的 Python 环境。依赖列表见 requirements.txt。整合仓库已包含特征缓存和六个 A/B checkpoint；WAV 可由根目录 scripts/download_crema.py 从官方来源下载。请从整合仓库根目录执行以下命令。

```bash
python3 hw3/level_tradeoff/experiment.py \
  --features data/features_K8.npz \
  --audio data/crema \
  --pilot hw3/runs \
  --out hw3/level_tradeoff/reproduction
python3 hw3/level_tradeoff/summarize.py \
  --out hw3/level_tradeoff/reproduction \
  --features data/features_K8.npz
```

实验拒绝覆盖已存在的输出目录；若重复运行，应另设 `--out`，并向汇总脚本传入同一个 `--out`。汇总脚本默认从 config.json 读取特征路径；迁移机器后可用 `--features` 指定新路径。本机结果由脚本实际执行得到，原 HW2、原 ZIP、原实验结果未覆盖。

## 如何理解这个对照

A、B 使用已经训练好的原始拼接和输入中心化模型。C 保留 B 的输入处理及按位置读出，增加 2→16→6 无偏置水平网络和 64→1 门控。评分门控只读相对表示。训练时对每条录音丢弃整个水平贡献，不做 inverted-dropout 放大；推理时水平贡献打开。

p 是训练干预，不是实际依赖大小。对每个 C 还执行固定相对分支、只移动水平输入的干预，并在验证集比较推理强度 lambda=0、0.25、0.5、0.75、1。lambda=0 的 C 与独立训练的 B 不等价。

±6 dB 对自然 log-RMS 的理想位移约为 ±0.6908，对当前 128-mel MFCC-0 的理想位移约为 ±67.8823；进入模型前分别除以训练标准差。该理想水平干预与真实波形增益重提取的结果分开保存。

## 限制

既往测试数据已启发类别关注和设计。本次只用验证集冻结 p 的选择，但测试结果仍属于同一数据集上的探索性后续证据。按种子的 SD 不是跨新说话人的置信区间。固定网格中的非单调结果和所有类别都需要报告。

真实波形增益检查复用干净 F0、裁剪和边界，并禁止削波，与前一轮 pilot 一致；不代表所有设备、噪声或说话人差异。中心化不是完全的说话人归一化。两个幅度相关特征并不完全等价，也不能一概声称使用水平信息必然导致 top-1 翻转。

AI 协助实现、运行和分析。本项目不声称双分支或通道丢弃是全新的方法；贡献是针对当前基线的假设、受控对照及可追溯结果。
