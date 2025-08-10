# RNN分类模型训练管道

这个项目实现了一个基于双向LSTM的RNN分类模型，用于区分人类和ChatGPT生成的文本。模型通过分析负对数似然分数序列来进行二分类。

## 项目结构

```
rnn/
├── data_loader.py      # 数据加载和处理
├── model.py           # 模型定义
├── trainer.py         # 训练器
├── train.py          # 主训练脚本
├── test_pipeline.py  # 管道测试脚本
└── README.md         # 说明文档
```

## 模型架构

### BiLSTMClassifier
- **输入**: 负对数似然分数序列 (每个时间步一个标量)
- **LSTM层**: 2层双向LSTM
- **隐藏层大小**: 可配置 (默认128)
- **分类头**: 3层MLP (ReLU + Dropout)
- **输出**: 2类分类 (人类 vs ChatGPT)

### SimpleLSTMClassifier
- 简化版LSTM模型，用于对比实验
- 单层单向LSTM
- 较小的隐藏层大小

## 数据格式

输入数据为文本文件，每行包含一个序列的负对数似然分数，用空格分隔：

```
11.13528 3.72001 10.88363 10.93882 0.16693 0.18061 5.67179 ...
9.04648 5.01506 2.77325 0.00815 8.66432 9.81500 4.27468 ...
...
```

- 人类数据文件: `poem_Human_llama3-8b-instruct.txt` (标签: 0)
- ChatGPT数据文件: `poem_ChatGPT_poem_llama3-8b-instruct.txt` (标签: 1)

## 安装依赖

```bash
pip install torch numpy matplotlib scikit-learn seaborn tqdm
```

## 快速开始

### 1. 测试管道

首先运行测试脚本确保所有组件正常工作：

```bash
cd rnn
python test_pipeline.py
```

### 2. 开始训练

使用默认参数进行训练：

```bash
python train.py
```

### 3. 自定义训练

```bash
python train.py \
    --batch_size 32 \
    --hidden_size 256 \
    --num_layers 3 \
    --learning_rate 0.0005 \
    --num_epochs 100 \
    --max_length 200 \
    --output_dir my_experiment
```

## 命令行参数

### 数据参数
- `--human_file`: 人类数据文件路径
- `--chatgpt_file`: ChatGPT数据文件路径
- `--max_length`: 序列最大长度 (None表示使用所有数据)
- `--batch_size`: 批次大小 (默认: 32)

### 模型参数
- `--model_type`: 模型类型 (`bilstm` 或 `simple`)
- `--hidden_size`: LSTM隐藏层大小 (默认: 128)
- `--num_layers`: LSTM层数 (默认: 2)
- `--dropout`: Dropout概率 (默认: 0.5)
- `--bidirectional`: 是否使用双向LSTM (默认: True)

### 训练参数
- `--num_epochs`: 训练轮数 (默认: 50)
- `--learning_rate`: 学习率 (默认: 0.001)
- `--weight_decay`: 权重衰减 (默认: 1e-5)
- `--patience`: 早停耐心值 (默认: 10)

### 其他参数
- `--device`: 训练设备 (默认: 自动选择)
- `--seed`: 随机种子 (默认: 42)
- `--output_dir`: 输出目录 (默认: outputs)
- `--save_plots`: 是否保存图表 (默认: True)

## 输出文件

训练完成后，会在输出目录中生成以下文件：

```
outputs/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   └── best_model_epoch_X.pth    # 最佳模型权重
├── results/
│   ├── metrics.json              # 评估指标
│   ├── predictions.npy           # 预测结果
│   ├── true_labels.npy           # 真实标签
│   └── probabilities.npy         # 预测概率
├── plots/
│   ├── training_history.png      # 训练历史图表
│   └── confusion_matrix.png      # 混淆矩阵
└── config.json                   # 训练配置
```

## 评估指标

模型会输出以下评估指标：

- **准确率 (Accuracy)**: 正确分类的样本比例
- **精确率 (Precision)**: 预测为正类中实际为正类的比例
- **召回率 (Recall)**: 实际正类中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **AUC**: ROC曲线下面积

## 示例结果

典型的训练结果：

```
============================================================
测试结果
============================================================
准确率: 0.9234
精确率: 0.9187
召回率: 0.9281
F1分数: 0.9234
AUC: 0.9456
============================================================
```

## 模型加载和推理

```python
import torch
from model import BiLSTMClassifier

# 加载模型
model = BiLSTMClassifier(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.5,
    bidirectional=True
)

# 加载权重
checkpoint = torch.load('checkpoints/best_model_epoch_X.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 推理
model.eval()
with torch.no_grad():
    # 准备输入数据
    sequences = torch.tensor([[...]])  # 你的数据
    lengths = torch.tensor([...])      # 序列长度
    
    outputs = model(sequences, lengths)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)
```

## 故障排除

### 常见问题

1. **内存不足**: 减小批次大小或序列长度
2. **训练过慢**: 使用GPU或减小模型大小
3. **过拟合**: 增加Dropout或减少模型复杂度
4. **欠拟合**: 增加模型复杂度或训练轮数

### 调试技巧

- 使用 `test_pipeline.py` 验证各个组件
- 从小数据集开始测试
- 检查数据文件路径是否正确
- 监控训练和验证损失曲线

## 扩展功能

### 添加新的模型架构

在 `model.py` 中添加新的模型类：

```python
class MyCustomModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 你的模型定义
    
    def forward(self, x, lengths=None):
        # 你的前向传播
        return output
```

### 添加新的评估指标

在 `trainer.py` 的 `evaluate` 方法中添加新的指标计算。

### 数据增强

在 `data_loader.py` 中实现数据增强技术，如序列截断、填充等。

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目采用MIT许可证。
