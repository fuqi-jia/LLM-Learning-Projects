# Transformer模型详细实现指南

## 目录
- [概述](#概述)
- [核心组件](#核心组件)
  - [1. 多头注意力机制](#1-多头注意力机制)
  - [2. 前馈神经网络](#2-前馈神经网络)
  - [3. 层归一化](#3-层归一化)
  - [4. 位置编码](#4-位置编码)
  - [5. Transformer块](#5-transformer块)
  - [6. 完整Transformer模型](#6-完整transformer模型)
- [数学公式详解](#数学公式详解)
- [运行方法](#运行方法)
- [模型配置](#模型配置)
- [扩展功能](#扩展功能)

## 概述

这是一个简易但完整的Transformer模型实现，基于《Attention Is All You Need》论文。该实现包含了Transformer的所有核心组件，可以用于语言建模、文本分类等自然语言处理任务。

### 主要特性
- ✅ 完整的多头注意力机制
- ✅ 位置编码
- ✅ 层归一化和残差连接
- ✅ 前馈神经网络
- ✅ 支持掩码机制
- ✅ 可配置的超参数
- ✅ 批量处理支持

## 核心组件

### 1. 多头注意力机制

多头注意力是Transformer的核心组件，允许模型同时关注序列中不同位置的信息。

#### 数学公式
对于输入序列，注意力机制计算如下：

**缩放点积注意力：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**多头注意力：**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### 代码实现要点
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # d_model: 模型维度 (通常为512)
        # n_heads: 注意力头数 (通常为8)
        # d_k: 每个头的维度 = d_model // n_heads
```

#### 关键步骤
1. **线性变换**: 将输入通过W^Q, W^K, W^V变换为Query, Key, Value
2. **多头分割**: 将Q, K, V分割为多个头
3. **缩放点积注意力**: 计算注意力权重和输出
4. **头部合并**: 将多个头的输出拼接
5. **输出投影**: 通过W^O进行最终线性变换

### 2. 前馈神经网络

前馈网络为Transformer提供非线性变换能力。

#### 数学公式
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

#### 结构特点
- 两层全连接网络
- 第一层：`d_model → d_ff` (通常d_ff = 4 * d_model)
- 激活函数：ReLU
- 第二层：`d_ff → d_model`
- Dropout正则化

### 3. 层归一化

层归一化用于稳定训练过程，在每个子层之后应用。

#### 数学公式
```
LayerNorm(x) = γ * (x - μ) / σ + β
其中:
μ = mean(x)  # 均值
σ = std(x)   # 标准差
γ, β 是可学习参数
```

#### 作用
- 标准化激活值分布
- 加速收敛
- 提高训练稳定性

### 4. 位置编码

由于Transformer缺乏递归结构，需要位置编码来理解序列顺序。

#### 数学公式
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中：
- `pos`: 位置索引
- `i`: 维度索引
- `d_model`: 模型维度

#### 特性
- 固定的正弦和余弦函数
- 允许模型学习相对位置关系
- 支持任意长度的序列

### 5. Transformer块

Transformer块是基本的构建单元，包含注意力和前馈网络。

#### 结构
```
x → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → output
    ↑_______________|                    ↑_______________|
    残差连接                             残差连接
```

#### 数学表示
```
# 子层1：多头注意力
x1 = LayerNorm(x + MultiHeadAttention(x))

# 子层2：前馈网络  
x2 = LayerNorm(x1 + FeedForward(x1))
```

### 6. 完整Transformer模型

完整模型将多个Transformer块堆叠，并添加输入/输出处理。

#### 架构流程
1. **输入嵌入**: 将token转换为向量
2. **位置编码**: 添加位置信息
3. **Transformer块**: N层堆叠的Transformer块
4. **层归一化**: 最终的层归一化
5. **输出投影**: 投影到词汇表大小

## 数学公式详解

### 注意力机制详细推导

1. **输入表示**
   ```
   X ∈ R^(seq_len × d_model)  # 输入序列
   ```

2. **线性变换**
   ```
   Q = XW^Q,  W^Q ∈ R^(d_model × d_model)
   K = XW^K,  W^K ∈ R^(d_model × d_model)  
   V = XW^V,  W^V ∈ R^(d_model × d_model)
   ```

3. **多头分割**
   ```
   Q_i = Q[:, i*d_k:(i+1)*d_k]  # 第i个头的Query
   K_i = K[:, i*d_k:(i+1)*d_k]  # 第i个头的Key
   V_i = V[:, i*d_k:(i+1)*d_k]  # 第i个头的Value
   ```

4. **注意力计算**
   ```
   Attention_i = softmax(Q_i K_i^T / √d_k) V_i
   ```

5. **头部拼接**
   ```
   MultiHead = Concat(Attention_1, ..., Attention_h) W^O
   ```

### 位置编码推导

位置编码使用正弦和余弦函数的组合：

```python
# 对于位置pos和维度i：
if i % 2 == 0:  # 偶数维度
    PE[pos][i] = sin(pos / 10000^(i/d_model))
else:           # 奇数维度  
    PE[pos][i] = cos(pos / 10000^((i-1)/d_model))
```

这种设计的优势：
- 每个位置有唯一的编码
- 相对位置关系可以通过三角恒等式学习
- 支持超出训练长度的序列

## 运行方法

### 1. 基本使用

```python
import torch
from transformer import Transformer, create_padding_mask

# 创建模型
model = Transformer(
    vocab_size=10000,  # 词汇表大小
    d_model=512,       # 模型维度
    n_heads=8,         # 注意力头数
    n_layers=6,        # Transformer层数
    d_ff=2048,         # 前馈网络隐藏层维度
    max_len=5000,      # 最大序列长度
    dropout=0.1        # Dropout率
)

# 准备输入数据
batch_size = 2
seq_len = 20
input_ids = torch.randint(0, 10000, (batch_size, seq_len))

# 创建掩码
padding_mask = create_padding_mask(input_ids)

# 前向传播
output = model(input_ids, padding_mask)
print(f"输出形状: {output.shape}")  # [batch_size, seq_len, vocab_size]
```

### 2. 训练示例

```python
import torch.optim as optim
import torch.nn as nn

# 初始化模型和优化器
model = Transformer(vocab_size=10000)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, target_ids = batch
        
        # 前向传播
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 3. 推理示例

```python
# 生成文本
model.eval()
with torch.no_grad():
    # 输入起始序列
    input_ids = torch.tensor([[1, 2, 3]])  # [batch_size, seq_len]
    
    for _ in range(50):  # 生成50个token
        outputs = model(input_ids)
        
        # 获取下一个token
        next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
        
        # 拼接到输入序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
    print("生成的序列:", input_ids)
```

## 模型配置

### 标准配置

| 配置 | Base | Large |
|------|------|-------|
| d_model | 512 | 1024 |
| n_heads | 8 | 16 |
| n_layers | 6 | 24 |
| d_ff | 2048 | 4096 |
| 参数量 | ~65M | ~340M |

### 自定义配置

```python
# 小型模型（用于实验）
small_config = {
    'vocab_size': 5000,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 1024,
    'dropout': 0.1
}

# 大型模型（用于生产）
large_config = {
    'vocab_size': 50000,
    'd_model': 1024,
    'n_heads': 16,
    'n_layers': 12,
    'd_ff': 4096,
    'dropout': 0.1
}
```

## 扩展功能

### 1. 掩码机制

#### 填充掩码
```python
def create_padding_mask(seq, pad_idx=0):
    """创建填充掩码，忽略填充token"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

#### 因果掩码（用于解码器）
```python
def generate_square_subsequent_mask(sz):
    """创建因果掩码，防止看到未来信息"""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf'))
```

### 2. 性能优化

#### 1. 梯度累积
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. 学习率调度
```python
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(step):
    warmup_steps = 4000
    return min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = LambdaLR(optimizer, lr_lambda)
```

### 3. 模型保存和加载

```python
# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
}, 'transformer_model.pth')

# 加载模型
checkpoint = torch.load('transformer_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 注意事项

1. **内存使用**: 注意力机制的复杂度为O(n²)，长序列会消耗大量内存
2. **梯度爆炸**: 建议使用梯度裁剪防止梯度爆炸
3. **学习率**: Transformer对学习率敏感，建议使用warmup策略
4. **数据预处理**: 确保正确的token化和填充处理

## 参考文献

- Vaswani, A., et al. "Attention is all you need." NIPS 2017.
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
