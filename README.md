# Taylor Swift 歌词生成器

这个项目使用RNN和LSTM网络来生成Taylor Swift风格的歌词。项目基于[Corpus of Taylor Swift](https://github.com/sagesolar/Corpus-of-Taylor-Swift)数据集。

## 项目结构

```
taylor_lyrics_generator/
├── logs/                # TensorBoard日志
├── checkpoints/         # 训练好的模型结果
│   ├── rnn/             # RNN模型
│   └── lstm/            # LSTM模型
├── utils/               
│   ├── data_utils.py    # 数据处理脚本
│   └── evaluation.py    # 评估指标脚本
├── models/              # 模型定义
│   ├── rnn_model.py     # RNN模型定义
│   └── lstm_model.py    # LSTM模型定义
├── environment.yml      # 环境配置文件
├── train.py             # 训练脚本
├── generate_lyrics.py   # 生成歌词脚本
└── compare_models.py    # 模型比较脚本
```

## 环境设置

使用Conda创建环境：

```bash
conda env create -f environment.yml
conda activate taylor_lyrics
```

## 数据准备

本项目使用Taylor Swift歌词语料库，数据来源于[Corpus of Taylor Swift](https://github.com/sagesolar/Corpus-of-Taylor-Swift)。需将此仓库克隆到与项目平行的目录中。

主要使用的数据文件是：
- `Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json`: 包含所有歌词的扁平化文件

## 训练模型

### 训练RNN模型

```bash
python train.py --model_type rnn --hidden_size 256 --num_layers 2 --dropout 0.5 --batch_size 64 --learning_rate 0.0001 --epochs 50 --sequence_length 100 --data_path ../Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json
```

### 训练LSTM模型

```bash
python train.py --model_type lstm --hidden_size 256 --num_layers 2 --dropout 0.5 --batch_size 32 --learning_rate 0.001 --epochs 50 --sequence_length 100 --bidirectional --data_path ../Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json
```

## 生成歌词

使用训练好的模型生成歌词：

```bash
python generate_lyrics.py --model_dir ./models/lstm/[实际文件夹] --max_length 1000 --temperatures 0.5 0.7 1.0 1.2
```

## 比较模型

比较RNN和LSTM模型的表现：

```bash
python compare_models.py --rnn_dir /root/Taylor/taylor_lyrics_generator/checkpoints/rnn/[实际文件夹] --lstm_dir /root/Taylor/taylor_lyrics_generator/checkpoints/lstm/[实际文件夹] --data_path ../Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json
```

## 评估指标

本项目使用以下指标评估模型性能：

1. **困惑度 (Perplexity)**: 衡量模型预测下一个字符的能力，越低越好
2. **多样性指标**:
   - char_diversity: 字符级多样性
   - bigram_diversity: 二元组多样性
   - trigram_diversity: 三元组多样性
3. **METEOR**: 衡量生成文本与参考文本的相似度
4. **分布对比**: 原始数据和生成数据的字符分布对比图

## 模型设计考量

### RNN模型

RNN模型采用基本的循环神经网络结构，主要特点包括：
- 使用He初始化权重，提高训练稳定性
- 对隐藏层到隐藏层的权重使用正交初始化，缓解梯度问题
- 在多层RNN中使用Dropout防止过拟合

### LSTM模型

LSTM模型相比RNN更加复杂，可以更好地捕获长距离依赖关系：
- 使用Xavier/Glorot初始化输入到隐藏状态的权重
- 对隐藏状态到隐藏状态的权重使用正交初始化
- 偏置的遗忘门部分初始化为1.0，帮助模型保留长期记忆
- 支持双向LSTM选项，可以同时考虑上下文信息

## TensorBoard可视化

训练过程中会记录以下信息到TensorBoard：
- 每个step的训练集和验证集损失
- 每个epoch的训练集和验证集困惑度
- 每个epoch的梯度信息
- 定期生成的示例文本

启动TensorBoard查看：

```bash
tensorboard --logdir=logs
```

## 实验结果

通过实验，我们发现LSTM模型在生成Taylor Swift风格歌词方面表现优于基本RNN模型，主要体现在：
- 更低的困惑度(Perplexity)
- 更高的文本多样性指标
- 生成的歌词更富有Taylor Swift的风格特点

具体模型推理结果可在各模型目录下的`eval_results.json`文件中查看。
- 包含测试集上的test_perplexity和文本多样性指标
- 模型生成文本分布与原始数据分布的JS散度

为了更直观的看到模型生成文本的分布，我们在各模型目录下保存了分布直方图的图片。
- `char_distribution.png`是数据集的真实字符分布
- `rnn_vs_original_distribution.png`是模型生成文本和数据集真实字符分布的对比图

在checkpoints文件夹下我们还保存了RNN和LSTM对比的详细数据，可在`model_comparison.json`中查看。


