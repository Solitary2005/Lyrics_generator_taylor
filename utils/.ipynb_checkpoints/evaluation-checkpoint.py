import numpy as np
import matplotlib.pyplot as plt
import torch
from nltk.translate import meteor_score
from collections import Counter
import math
from scipy.spatial.distance import jensenshannon
import nltk


def try_download_nltk_data():
    """尝试下载NLTK所需数据并设置路径"""
    try:
        import nltk
        nltk.data.path.append('/root/nltk_data')  # 确保路径正确
        nltk.download('wordnet', download_dir='/root/nltk_data')
        nltk.download('punkt', download_dir='/root/nltk_data')
        nltk.download('punkt_tab', download_dir='/root/nltk_data')
    except Exception as e:
        print(f"警告：无法下载NLTK数据，如果已下载则忽略此警告: {e}")


def calculate_perplexity(model, data_loader, device):
    """计算困惑度(Perplexity)
    
    困惑度 = 2^交叉熵，是语言模型常用的评估指标，越低越好
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备类型
        
    Returns:
        float: 困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            output = model(inputs)
            
            # 调整输出和目标的形状以计算损失
            output = output.view(-1, output.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(output, targets)
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    # 计算平均损失
    avg_loss = total_loss / total_tokens
    # 计算困惑度
    perplexity = math.exp(avg_loss)
    
    return perplexity


def calculate_diversity(generated_text):
    """计算生成文本的多样性指标
    
    Args:
        generated_text (str): 生成的文本
        
    Returns:
        dict: 多样性指标字典
    """
    # 计算字符级多样性
    chars = list(generated_text)
    char_counter = Counter(chars)
    char_diversity = len(char_counter) / len(chars) if chars else 0
    
    # 计算二元组多样性
    bigrams = [generated_text[i:i+2] for i in range(len(generated_text)-1)]
    bigram_counter = Counter(bigrams)
    bigram_diversity = len(bigram_counter) / len(bigrams) if bigrams else 0
    
    # 计算三元组多样性
    trigrams = [generated_text[i:i+3] for i in range(len(generated_text)-2)]
    trigram_counter = Counter(trigrams)
    trigram_diversity = len(trigram_counter) / len(trigrams) if trigrams else 0
    
    return {
        'char_diversity': char_diversity,
        'bigram_diversity': bigram_diversity,
        'trigram_diversity': trigram_diversity
    }


def calculate_meteor(reference, hypothesis):
    """计算METEOR评分
    
    METEOR (Metric for Evaluation of Translation with Explicit ORdering)是一种评估生成文本与参考文本相似度的指标
    
    Args:
        reference (str): 参考文本
        hypothesis (str): 生成的文本
        
    Returns:
        float: METEOR评分
    """
    try_download_nltk_data()
    
    # 将文本分词
    ref_tokens = nltk.word_tokenize(reference.lower(), language='english')
    hyp_tokens = nltk.word_tokenize(hypothesis.lower(), language='english')
    
    # 计算METEOR评分
    score = meteor_score.meteor_score([ref_tokens], hyp_tokens)
    return score


def calculate_char_distribution(text):
    """计算文本中字符的分布
    
    Args:
        text (str): 文本
        
    Returns:
        dict: 字符分布字典
    """
    chars = list(text)
    char_counter = Counter(chars)
    total_chars = len(chars)
    
    # 计算每个字符的概率
    char_distribution = {char: count / total_chars for char, count in char_counter.items()}
    return char_distribution


def calculate_js_divergence(dist1, dist2):
    """计算两个分布之间的Jensen-Shannon散度
    
    Args:
        dist1 (dict): 第一个分布
        dist2 (dict): 第二个分布
        
    Returns:
        float: Jensen-Shannon散度
    """
    # 合并两个分布的键
    all_keys = set(list(dist1.keys()) + list(dist2.keys()))
    
    # 创建分布向量
    vec1 = np.array([dist1.get(k, 0) for k in all_keys])
    vec2 = np.array([dist2.get(k, 0) for k in all_keys])
    
    # 计算JS散度
    return jensenshannon(vec1, vec2)


def plot_distribution_comparison(original_dist, generated_dist, title="char distribution comparison"):
    """可视化原始数据和生成数据的概率分布对比
    
    Args:
        original_dist (dict): 原始分布
        generated_dist (dict): 生成分布
        title (str): 图表标题
    """
    # 合并两个分布的键并排序
    all_chars = sorted(set(list(original_dist.keys()) + list(generated_dist.keys())))
    
    # 提取分布值
    original_values = [original_dist.get(char, 0) for char in all_chars]
    generated_values = [generated_dist.get(char, 0) for char in all_chars]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置条形宽度和位置
    width = 0.35
    x = np.arange(len(all_chars))
    
    # 绘制条形图
    ax.bar(x - width/2, original_values, width, label='original data')
    ax.bar(x + width/2, generated_values, width, label='generated data')
    
    # 设置图表标签和标题
    ax.set_xlabel('char')
    ax.set_ylabel('probability')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_chars, rotation=90)
    ax.legend()
    
    # 调整布局
    fig.tight_layout()
    
    # 返回图表对象以便保存
    return fig