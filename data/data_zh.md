# 数学加法数据生成器

## 目录
- [概述](#概述)
- [功能特性](#功能特性)
- [数据格式](#数据格式)
- [使用方法](#使用方法)
- [配置参数](#配置参数)
- [与Transformer集成](#与transformer集成)
- [数据分析](#数据分析)
- [扩展功能](#扩展功能)

## 概述

这是一个专门为训练Transformer模型设计的数学加法数据生成器。它可以自动生成大量的加法运算数据，用于训练序列到序列的模型学习基本的数学运算能力。

### 核心功能
- ✅ 随机生成多项加法表达式
- ✅ 自动计算正确答案
- ✅ 构建完整的词汇表
- ✅ 数据集自动划分（训练/验证/测试）
- ✅ 支持可配置的数据规模和复杂度
- ✅ 与Transformer模型无缝集成

## 功能特性

### 1. 灵活的数据生成

```python
# 生成示例
表达式: "23+45+12"
结果: 80
输入序列: "<START>23+45+12="
输出序列: "80<END>"
```

#### 可配置参数
- **加数个数**: 2-5个数字（可配置）
- **数字范围**: 1-100（可配置）
- **样本数量**: 任意数量
- **随机种子**: 确保结果可重现

### 2. 完整的词汇表设计

```json
{
  "<PAD>": 0,    // 填充token
  "<START>": 1,  // 开始token
  "<END>": 2,    // 结束token
  "+": 3,        // 加号
  "=": 4,        // 等号
  "0": 5,        // 数字0-9
  "1": 6,
  ...
  "9": 14
}
```

#### 词汇表特点
- 总共15个token
- 支持任意多位数字（通过单个数字组合）
- 包含必要的数学运算符号
- 特殊token用于序列标记

### 3. 智能数据处理

#### 分词处理
```python
# 输入: "123+45="
# 分词结果: ['1', '2', '3', '+', '4', '5', '=']
# Token IDs: [6, 7, 8, 3, 9, 10, 4]
```

#### 避免重复
- 自动检测重复表达式
- 确保数据集的多样性
- 提高训练效果

### 4. 数据集划分

默认划分比例：
- **训练集**: 80%
- **验证集**: 10% 
- **测试集**: 10%

支持自定义划分比例。

## 数据格式

### 训练样例结构

```json
{
  "input_ids": [1, 6, 7, 3, 9, 10, 4],           // 输入token序列
  "output_ids": [11, 8, 2],                      // 输出token序列
  "expression": "12+45",                          // 原始表达式
  "result": 57,                                   // 计算结果
  "input_text": "<START>12+45=",                 // 输入文本
  "output_text": "57<END>"                       // 输出文本
}
```

### 文件结构

```
data_output/
├── train.json      # 训练集
├── val.json        # 验证集
├── test.json       # 测试集
└── vocab.json      # 词汇表
```

## 使用方法

### 1. 基本使用

```bash
# 生成默认数据集（10,000个样本）
python data/generate.py

# 生成指定数量的样本
python data/generate.py --num_samples 50000

# 指定输出目录
python data/generate.py --output_dir ./my_data
```

### 2. 高级配置

```bash
# 完整配置示例
python data/generate.py \
    --num_samples 100000 \
    --max_numbers 6 \
    --max_value 999 \
    --min_numbers 2 \
    --output_dir ./large_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

### 3. 编程接口

```python
from data.generate import MathDataGenerator

# 创建生成器
generator = MathDataGenerator(
    max_numbers=5,
    max_value=100,
    min_numbers=2
)

# 生成数据集
dataset = generator.generate_dataset(1000)

# 划分数据集
train_set, val_set, test_set = generator.split_dataset(dataset)

# 保存数据
generator.save_dataset(train_set, 'train.json')
generator.save_vocab('vocab.json')
```

## 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_samples` | 10000 | 生成样本总数 |
| `--max_numbers` | 5 | 每个表达式最大加数个数 |
| `--max_value` | 100 | 数字最大值 |
| `--min_numbers` | 2 | 每个表达式最小加数个数 |
| `--output_dir` | `./data_output` | 输出目录 |
| `--train_ratio` | 0.8 | 训练集比例 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--test_ratio` | 0.1 | 测试集比例 |
| `--seed` | 42 | 随机种子 |

### 配置示例

#### 小规模数据集（适合快速实验）
```bash
python data/generate.py \
    --num_samples 1000 \
    --max_numbers 3 \
    --max_value 50
```

#### 大规模数据集（适合完整训练）
```bash
python data/generate.py \
    --num_samples 500000 \
    --max_numbers 8 \
    --max_value 999
```

#### 简单数据集（适合调试）
```bash
python data/generate.py \
    --num_samples 100 \
    --max_numbers 2 \
    --max_value 10
```

## 与Transformer集成

### 1. 数据加载器

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

class MathDataset(Dataset):
    def __init__(self, data_path, max_length=20):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 输入序列填充
        input_ids = item['input_ids'][:self.max_length]
        input_ids += [0] * (self.max_length - len(input_ids))
        
        # 输出序列填充
        output_ids = item['output_ids'][:self.max_length]
        output_ids += [0] * (self.max_length - len(output_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }

# 使用示例
train_dataset = MathDataset('data_output/train.json')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 2. 训练脚本示例

```python
import torch
import torch.nn as nn
from transformer import Transformer

# 加载词汇表
with open('data_output/vocab.json', 'r') as f:
    vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']

# 创建模型
model = Transformer(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_len=50
)

# 训练设置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD token

# 训练循环
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        
        # 前向传播
        outputs = model(input_ids)
        
        # 计算损失
        loss = criterion(
            outputs.view(-1, vocab_size), 
            output_ids.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

### 3. 推理示例

```python
def predict_math(model, generator, expression):
    """预测数学表达式的结果"""
    model.eval()
    
    # 构建输入
    input_text = f"<START>{expression}="
    input_ids = generator.tokenize(input_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        # 生成输出
        output_ids = []
        current_input = input_tensor
        
        for _ in range(10):  # 最大生成长度
            outputs = model(current_input)
            next_token = torch.argmax(outputs[0, -1, :])
            
            if next_token.item() == generator.vocab['<END>']:
                break
                
            output_ids.append(next_token.item())
            current_input = torch.cat([
                current_input, 
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
        
        # 解码结果
        result_text = generator.detokenize(output_ids)
        return result_text

# 使用示例
expression = "25+37+14"
predicted_result = predict_math(model, generator, expression)
print(f"{expression} = {predicted_result}")
```

## 数据分析

### 1. 数据集统计

生成器会自动提供详细的统计信息：

```
=== 训练集统计 ===
数据集大小: 8000
输入序列长度 - 最小: 5, 最大: 15, 平均: 9.14
输出序列长度 - 最小: 2, 最大: 3, 平均: 2.77

示例数据:
示例 1:
  表达式: 92+44+27
  结果: 163
  输入: <START>92+44+27=
  输出: 163<END>
```

### 2. 复杂度分析

#### 序列长度分布
- **最短输入**: `<START>1+2=` (5个token)
- **最长输入**: `<START>99+98+97+96+95=` (15个token)
- **平均长度**: 约9个token

#### 数值范围
- **最小结果**: 2 (1+1)
- **最大结果**: 500 (5×100)
- **结果分布**: 相对均匀

### 3. 难度级别

```python
# 简单级别
--max_numbers 2 --max_value 10
# 示例: 3+7=10

# 中等级别
--max_numbers 4 --max_value 50  
# 示例: 23+31+45+12=111

# 困难级别
--max_numbers 6 --max_value 999
# 示例: 234+567+890+123+456+789=3059
```

## 扩展功能

### 1. 支持其他运算

```python
class ExtendedMathGenerator(MathDataGenerator):
    def __init__(self, operations=['+'], **kwargs):
        super().__init__(**kwargs)
        self.operations = operations
        
    def generate_expression(self):
        # 扩展支持减法、乘法等
        operation = random.choice(self.operations)
        if operation == '+':
            return super().generate_expression()
        elif operation == '-':
            # 实现减法逻辑
            pass
        elif operation == '*':
            # 实现乘法逻辑
            pass
```

### 2. 数据增强

```python
def augment_dataset(dataset, factor=2):
    """数据增强：重新排列加数顺序"""
    augmented = []
    for item in dataset:
        numbers = item['expression'].split('+')
        for _ in range(factor):
            random.shuffle(numbers)
            new_expr = '+'.join(numbers)
            # 创建新样本...
    return augmented
```

### 3. 验证工具

```python
def validate_dataset(dataset):
    """验证数据集的正确性"""
    errors = 0
    for item in dataset:
        expression = item['expression']
        expected = item['result']
        actual = eval(expression)  # 仅用于验证
        
        if actual != expected:
            errors += 1
            print(f"错误: {expression} = {expected}, 实际应为 {actual}")
    
    print(f"验证完成，错误数: {errors}/{len(dataset)}")
```

### 4. 可视化分析

```python
import matplotlib.pyplot as plt

def plot_length_distribution(dataset):
    """绘制序列长度分布"""
    lengths = [len(item['input_ids']) for item in dataset]
    plt.hist(lengths, bins=20)
    plt.xlabel('序列长度')
    plt.ylabel('频次')
    plt.title('输入序列长度分布')
    plt.show()

def plot_result_distribution(dataset):
    """绘制结果分布"""
    results = [item['result'] for item in dataset]
    plt.hist(results, bins=50)
    plt.xlabel('计算结果')
    plt.ylabel('频次')
    plt.title('加法结果分布')
    plt.show()
```

## 最佳实践

### 1. 数据集大小建议

| 模型规模 | 建议样本数 | 参数配置 |
|----------|------------|----------|
| 小型实验 | 1,000-10,000 | `--max_numbers 3 --max_value 50` |
| 中型训练 | 50,000-100,000 | `--max_numbers 5 --max_value 100` |
| 大型训练 | 500,000+ | `--max_numbers 8 --max_value 999` |

### 2. 训练建议

- **学习率**: 从1e-4开始，根据收敛情况调整
- **批大小**: 32-128，根据显存调整
- **序列长度**: 建议设置为数据集最大长度的1.2倍
- **早停机制**: 监控验证集损失，防止过拟合

### 3. 评估指标

```python
def calculate_accuracy(model, test_loader):
    """计算完全匹配准确率"""
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            predictions = model.generate(batch['input_ids'])
            targets = batch['output_ids']
            
            # 比较序列是否完全匹配
            for pred, target in zip(predictions, targets):
                if torch.equal(pred, target):
                    correct += 1
                total += 1
    
    return correct / total
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少样本数量
   - 降低序列最大长度
   - 使用较小的批大小

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据格式正确性
   - 增加模型容量

3. **生成结果错误**
   - 检查词汇表映射
   - 验证分词逻辑
   - 确认特殊token处理

### 调试技巧

```python
# 打印样本检查
for i, item in enumerate(dataset[:5]):
    print(f"样本 {i}:")
    print(f"  原始: {item['expression']} = {item['result']}")
    print(f"  输入: {item['input_text']}")
    print(f"  输出: {item['output_text']}")
    print(f"  验证: {eval(item['expression'])} == {item['result']}")
    print()
```

## 总结

这个数学加法数据生成器为Transformer模型提供了一个理想的学习环境：

- **数据质量高**: 所有样本都经过验证
- **格式标准**: 完全兼容序列到序列任务
- **可扩展性**: 易于扩展到其他数学运算
- **易于使用**: 简单的命令行接口和编程接口

通过这个工具，您可以快速生成高质量的训练数据，让Transformer模型学会基本的数学运算能力，为更复杂的推理任务打下基础。
