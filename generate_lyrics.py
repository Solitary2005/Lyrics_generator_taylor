import os
import sys
import torch
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_model import CharRNN
from models.lstm_model import CharLSTM
from utils.data_utils import generate_text, set_seed


def generate_from_model(model_dir, seed_texts, max_length=1000, temperature_range=None, output_dir=None):
    """使用训练好的模型生成文本
    
    Args:
        model_dir: 模型目录
        seed_texts: 种子文本列表
        max_length: 生成文本的最大长度
        temperature_range: 温度范围，默认为[0.5, 0.7, 1.0, 1.2]
        output_dir: 输出目录，默认为模型目录下的generate_{timestamp}
    """
    set_seed(42)
    
    # 如果没有提供温度范围，则使用默认值
    if temperature_range is None:
        temperature_range = [0.5, 0.7, 1.0, 1.2]
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join(model_dir, f'generate_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    

    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)
    
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
    
    # 加载权重
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 生成文本
    results = {}
    for seed_text in seed_texts:
        seed_results = {}
        for temperature in temperature_range:
            print(f'\nGenerating with seed: "{seed_text}", temperature: {temperature}')
            generated_text = generate_text(
                model=model,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                seed_text=seed_text,
                max_length=max_length,
                temperature=temperature,
                device=device
            )
            print(f'Generated: "{generated_text[:100]}..."')
            seed_results[str(temperature)] = generated_text
        results[seed_text] = seed_results
    
    with open(os.path.join(output_dir, 'generated_texts.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 创建一个漂亮的HTML
    html_content = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="UTF-8">',
        f'    <title>{model_config["model_type"].upper()} 生成的Taylor Swift风格歌词</title>',
        '    <style>',
        '        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }',
        '        h1 { color: #df2f55; }',  # 红
        '        h2 { color: #1db954; margin-top: 30px; }',  # 绿
        '        .seed { font-weight: bold; color: #0066cc; }',
        '        .lyrics { white-space: pre-wrap; background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }',
        '        .temperature { font-style: italic; color: #666; }',
        '    </style>',
        '</head>',
        '<body>',
        f'    <h1>{model_config["model_type"].upper()} 生成的Taylor Swift风格歌词</h1>',
        f'    <p>模型类型: {model_config["model_type"]}, 隐藏层大小: {model_config["hidden_size"]}, 层数: {model_config["num_layers"]}</p>',
    ]
    
    for seed_text, seed_results in results.items():
        html_content.append(f'    <h2>种子文本: <span class="seed">"{seed_text}"</span></h2>')
        for temperature, generated_text in seed_results.items():
            html_content.append(f'    <p class="temperature">温度: {temperature}</p>')
            html_content.append(f'    <div class="lyrics">{generated_text}</div>')
    
    html_content.extend([
        '</body>',
        '</html>'
    ])
    
    with open(os.path.join(output_dir, 'generated_texts.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
    
    print(f'\nResults saved to {output_dir}')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate text from a trained model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated texts')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum length of generated text per seed')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5, 0.7, 1.0, 1.2],
                        help='List of temperature values for sampling')
    
    args = parser.parse_args()
    
    # 定义种子文本
    seed_texts = [
        "I love you",
        "When I was young",
        "All I want is",
        "She said to me",
        "In my dreams",
        "Look what you made me do",
        "I knew you were trouble",
        "We are never ever",
        "Today was a fairytale"
    ]
    
    # 生成文本
    generate_from_model(
        model_dir=args.model_dir,
        seed_texts=seed_texts,
        max_length=args.max_length,
        temperature_range=args.temperatures,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
