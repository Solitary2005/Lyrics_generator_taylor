import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random


def set_seed(seed):
    """设置所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_lyrics_data(file_path):
    """加载歌词数据并清理多余字符
    
    Args:
        file_path (str): 歌词JSON文件路径
    
    Returns:
        str: 合并并清理后的歌词文本
    """
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        lyrics_data = json.load(file)
    
    if isinstance(lyrics_data, dict):  # flat-song-lyrics.json
        lyrics_text = '\n'.join(value.strip('"') for value in lyrics_data.values())
    elif isinstance(lyrics_data, list):  # album-song-lyrics.json
        all_lyrics = []
        for album in lyrics_data:
            for song in album.get('Songs', []):
                song_lyrics = [line.get('Text', '').strip('"') for line in song.get('Lyrics', [])]
                all_lyrics.append('\n'.join(song_lyrics))
        lyrics_text = '\n\n'.join(all_lyrics)
    
    # 清理多余字符
    lyrics_text = lyrics_text.replace('"', '').strip()
    
    return lyrics_text


def preprocess_text(text):
    """对文本进行预处理
    
    Args:
        text (str): 原始文本
        
    Returns:
        str: 预处理后的文本
    """
    # 转换为小写
    text = text.lower()
    
    # 保留一定的标点符号，因为它们对歌词生成很重要
    # 可以根据需要调整预处理的程度
    
    return text


def create_char_mappings(text):
    """创建字符到索引和索引到字符的映射
    
    Args:
        text (str): 预处理后的文本
    
    Returns:
        tuple: (char_to_idx, idx_to_char, vocab_size)
    """
    # 获取所有唯一字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # 创建字符到索引的映射
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    # 创建索引到字符的映射
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return char_to_idx, idx_to_char, vocab_size


def split_data(text, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """将文本数据分割为训练集、验证集和测试集
    
    Args:
        text (str): 完整文本
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
    
    Returns:
        tuple: (train_text, val_text, test_text)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 将文本分割成行
    lines = text.split('\n')
    random.shuffle(lines)
    
    train_size = int(len(lines) * train_ratio)
    val_size = int(len(lines) * val_ratio)
    
    train_text = '\n'.join(lines[:train_size])
    val_text = '\n'.join(lines[train_size:train_size + val_size])
    test_text = '\n'.join(lines[train_size + val_size:])
    
    return train_text, val_text, test_text


class LyricsDataset(Dataset):
    """歌词数据集类"""
    
    def __init__(self, text, char_to_idx, sequence_length=100):
        """初始化数据集
        
        Args:
            text (str): 歌词文本
            char_to_idx (dict): 字符到索引的映射
            sequence_length (int): 序列长度
        """
        self.text = text
        self.char_to_idx = char_to_idx
        self.sequence_length = sequence_length
        self.text_encoded = [char_to_idx[ch] for ch in text if ch in char_to_idx]
        self.total_sequences = len(self.text_encoded) - sequence_length
        
    def __len__(self):
        return max(0, self.total_sequences)
    
    def __getitem__(self, idx):
        # 获取输入序列
        sequence = self.text_encoded[idx:idx + self.sequence_length]
        # 获取目标序列（下一个字符）
        target = self.text_encoded[idx + 1:idx + self.sequence_length + 1]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def get_data_loaders(train_text, val_text, test_text, char_to_idx, 
                      sequence_length=100, batch_size=64):
    """创建数据加载器
    
    Args:
        train_text (str): 训练集文本
        val_text (str): 验证集文本
        test_text (str): 测试集文本
        char_to_idx (dict): 字符到索引的映射
        sequence_length (int): 序列长度
        batch_size (int): 批次大小
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = LyricsDataset(train_text, char_to_idx, sequence_length)
    val_dataset = LyricsDataset(val_text, char_to_idx, sequence_length)
    test_dataset = LyricsDataset(test_text, char_to_idx, sequence_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=500, temperature=1.0, device='cuda'):
    """使用训练好的模型生成文本
    
    Args:
        model: 训练好的模型
        char_to_idx (dict): 字符到索引的映射
        idx_to_char (dict): 索引到字符的映射
        seed_text (str): 种子文本
        max_length (int): 生成文本的最大长度
        temperature (float): 采样温度，越高多样性越大
        device (str): 设备类型
        
    Returns:
        str: 生成的文本
    """
    model.eval()
    
    # 将种子文本转换为索引序列
    current_text = seed_text.lower()
    encoded_input = [char_to_idx[ch] for ch in current_text if ch in char_to_idx]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 转换为张量
            x = torch.tensor([encoded_input[-model.sequence_length:]], dtype=torch.long).to(device)
            
            # 如果输入长度不足，进行填充
            if x.size(1) < model.sequence_length:
                pad_length = model.sequence_length - x.size(1)
                padding = torch.zeros(1, pad_length, dtype=torch.long).to(device)
                x = torch.cat([padding, x], dim=1)
            
            # 前向传播，获取最后一个时间步的输出
            output = model(x)
            output = output[0, -1, :].squeeze()
            
            # 应用温度参数
            output = output / temperature
            
            # 转换为概率分布
            probs = torch.softmax(output, dim=0)
            
            # 按概率分布采样下一个字符
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # 将新字符添加到结果中
            current_text += idx_to_char[next_char_idx]
            encoded_input.append(next_char_idx)
    
    return current_text[len(seed_text):]