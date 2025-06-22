import os
import sys
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate import meteor_score
import nltk


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_model import CharRNN
from models.lstm_model import CharLSTM
from utils.data_utils import load_lyrics_data, preprocess_text, create_char_mappings, generate_text
from utils.evaluation import calculate_perplexity, calculate_diversity, calculate_meteor, calculate_char_distribution, plot_distribution_comparison


def load_model_and_mappings(model_dir, device):
    """加载模型和字符映射
    
    Args:
        model_dir: 模型目录
        device: 设备
        
    Returns:
        tuple: (model, char_to_idx, idx_to_char)
    """
    # 加载模型配置
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)
    
    # 加载字符映射
    with open(os.path.join(model_dir, 'char_mappings.json'), 'r') as f:
        mappings = json.load(f)
    
    char_to_idx = {k: int(v) for k, v in mappings['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}
    vocab_size = mappings['vocab_size']
    
    # 创建模型
    if model_config['model_type'] == 'rnn':
        model = CharRNN(
            input_size=vocab_size,
            hidden_size=model_config['hidden_size'],
            output_size=vocab_size,
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            sequence_length=model_config['sequence_length']
        )
    elif model_config['model_type'] == 'lstm':
        model = CharLSTM(
            input_size=vocab_size,
            hidden_size=model_config['hidden_size'],
            output_size=vocab_size,
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            bidirectional=model_config.get('bidirectional', False),
            sequence_length=model_config['sequence_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['model_type']}")
    
    # 加载模型权重
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, char_to_idx, idx_to_char


def compare_models(rnn_dir, lstm_dir, original_data_path, seed_texts, max_length=500, temperature=1.0):
    """比较RNN和LSTM模型的性能
    
    Args:
        rnn_dir (str): RNN模型目录
        lstm_dir (str): LSTM模型目录
        original_data_path (str): 原始数据路径
        seed_texts (list): 种子文本列表
        max_length (int): 生成文本的最大长度
        temperature (float): 采样温度
    """
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载原始数据
    lyrics_text = load_lyrics_data(original_data_path)
    preprocessed_text = preprocess_text(lyrics_text)
    
    # 加载RNN模型
    rnn_model, rnn_char_to_idx, rnn_idx_to_char = load_model_and_mappings(rnn_dir, device)
    
    # 加载LSTM模型
    lstm_model, lstm_char_to_idx, lstm_idx_to_char = load_model_and_mappings(lstm_dir, device)
    
    # 生成文本进行比较
    rnn_generated_texts = []
    lstm_generated_texts = []
    
    print("Generating texts for comparison...")
    for seed_text in seed_texts:
        # 使用RNN生成文本
        rnn_text = generate_text(
            model=rnn_model,
            char_to_idx=rnn_char_to_idx,
            idx_to_char=rnn_idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        rnn_generated_texts.append(rnn_text)
        
        # 使用LSTM生成文本
        lstm_text = generate_text(
            model=lstm_model,
            char_to_idx=lstm_char_to_idx,
            idx_to_char=lstm_idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        lstm_generated_texts.append(lstm_text)
        
        print(f'\n--- Seed: "{seed_text}" ---')
        print(f'RNN: "{rnn_text[:100]}..."')
        print(f'LSTM: "{lstm_text[:100]}..."')
    
    # 合并所有生成的文本
    all_rnn_text = "\n\n".join(rnn_generated_texts)
    all_lstm_text = "\n\n".join(lstm_generated_texts)
    
    # 计算多样性指标
    rnn_diversity = calculate_diversity(all_rnn_text)
    lstm_diversity = calculate_diversity(all_lstm_text)
    
    print("\n--- Diversity Metrics ---")
    print(f'RNN: {rnn_diversity}')
    print(f'LSTM: {lstm_diversity}')
    
    # 计算METEOR评分（使用原始文本的一部分作为参考）
    # 从原始文本中随机选择一些行作为参考
    reference_lines = preprocessed_text.split('\n')
    np.random.shuffle(reference_lines)
    reference_text = '\n'.join(reference_lines[:100])  # 使用100行作为参考
    
    rnn_meteor = calculate_meteor(reference_text, all_rnn_text[:len(reference_text)])
    lstm_meteor = calculate_meteor(reference_text, all_lstm_text[:len(reference_text)])
    
    print("\n--- METEOR Scores ---")
    print(f'RNN: {rnn_meteor:.4f}')
    print(f'LSTM: {lstm_meteor:.4f}')
    
    # 计算字符分布
    original_dist = calculate_char_distribution(preprocessed_text)
    rnn_dist = calculate_char_distribution(all_rnn_text)
    lstm_dist = calculate_char_distribution(all_lstm_text)
    
    # 绘制并保存RNN的分布对比图
    rnn_fig = plot_distribution_comparison(original_dist, rnn_dist, title="char distribution comparison: RNN vs Original")
    rnn_fig.savefig(os.path.join(rnn_dir, 'rnn_vs_original_distribution.png'))
    
    # 绘制并保存LSTM的分布对比图
    lstm_fig = plot_distribution_comparison(original_dist, lstm_dist, title="char distribution comparison: LSTM vs Original")
    lstm_fig.savefig(os.path.join(lstm_dir, 'lstm_vs_original_distribution.png'))
    
    # 将结果保存到文件
    comparison_results = {
        'diversity': {
            'rnn': rnn_diversity,
            'lstm': lstm_diversity,
        },
        'meteor': {
            'rnn': rnn_meteor,
            'lstm': lstm_meteor,
        },
        'generated_samples': {
            'seeds': seed_texts,
            'rnn': rnn_generated_texts,
            'lstm': lstm_generated_texts,
        }
    }
    
    # 保存比较结果
    with open(os.path.join(os.path.dirname(rnn_dir), 'model_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\nComparison results saved to {os.path.join(os.path.dirname(rnn_dir), 'model_comparison.json')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Compare RNN and LSTM models')
    
    # 参数
    parser.add_argument('--rnn_dir', type=str, required=True,
                        help='Directory containing the RNN model')
    parser.add_argument('--lstm_dir', type=str, required=True,
                        help='Directory containing the LSTM model')
    parser.add_argument('--data_path', type=str, default='../Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json',
                        help='Path to the original lyrics data file')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    
    args = parser.parse_args()
    
    # 定义种子文本
    seed_texts = [
        "I love you", 
        "When I was", 
        "All I want", 
        "She said", 
        "In my dreams"
    ]
    
    # 比较模型
    compare_models(
        rnn_dir=args.rnn_dir,
        lstm_dir=args.lstm_dir,
        original_data_path=args.data_path,
        seed_texts=seed_texts,
        max_length=args.max_length,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()
