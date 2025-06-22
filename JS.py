import torch
from utils.evaluation import calculate_char_distribution, calculate_js_divergence
from utils.data_utils import preprocess_text, load_lyrics_data, create_char_mappings, generate_text
from models.rnn_model import CharRNN
from models.lstm_model import CharLSTM
import os
import json

def evaluate_js_divergence(model_type, model_path, data_path, seed_texts, max_length=500, temperature=0.8):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据并预处理
    lyrics_text = load_lyrics_data(data_path)
    preprocessed_text = preprocess_text(lyrics_text)
    char_to_idx, idx_to_char, _ = create_char_mappings(preprocessed_text)

    # 加载模型配置
    config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # 初始化模型
    if model_config['model_type'] == 'rnn':
        model = CharRNN(
            input_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['vocab_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            sequence_length=model_config['sequence_length']
        )
    elif model_config['model_type'] == 'lstm':
        model = CharLSTM(
            input_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['vocab_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional'],
            sequence_length=model_config['sequence_length']
        )
    else:
        raise ValueError("Unsupported model type")

    # 加载模型权重并移动到设备
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 生成文本
    generated_text = ""
    for seed_text in seed_texts:
        generated_text += generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=temperature,
            device=device  # 确保输入数据也在同一设备上
        )

    # 计算分布
    original_dist = calculate_char_distribution(preprocessed_text)
    generated_dist = calculate_char_distribution(generated_text)

    # 计算JS散度
    js_divergence = calculate_js_divergence(original_dist, generated_dist)
    print(f"Jensen-Shannon Divergence: {js_divergence}")
    js_div = {
        'js_divergence': js_divergence,
        'original_distribution': original_dist,
        'generated_distribution': generated_dist
    }
    with open(os.path.join(os.path.dirname(model_path), 'eval_results.json'), 'w') as f:
        json.dump(js_div, f)

if __name__ == "__main__":
    # 评估LSTM模型的JS散度
    evaluate_js_divergence(
        model_type="lstm",  # 或 "rnn"
        model_path="taylor_lyrics_generator/checkpoints/lstm/20250529-213819/best_model.pt",
        data_path="Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json",
        seed_texts=["I love you", "When I was", "All I want", "She said"]
    )
    # 评估RNN模型的JS散度
    evaluate_js_divergence(
        model_type="rnn",
        model_path="taylor_lyrics_generator/checkpoints/rnn/20250529-210554/best_model.pt",
        data_path="Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json",
        seed_texts=["I love you", "When I was", "All I want", "She said"]
    )