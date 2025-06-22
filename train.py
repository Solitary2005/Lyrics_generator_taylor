import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_model import CharRNN
from models.lstm_model import CharLSTM
from utils.data_utils import load_lyrics_data, preprocess_text, create_char_mappings, split_data, get_data_loaders, generate_text, set_seed
from utils.evaluation import calculate_perplexity, calculate_diversity, calculate_char_distribution, plot_distribution_comparison, calculate_js_divergence


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                epochs, log_dir, model_dir, idx_to_char, char_to_idx, 
                generate_every=10, save_every=10, seed_texts=None, print_gradients=True):
    """训练模型
    
    Args:
        model
        train_loader
        val_loader
        optimizer
        criterion
        device
        epochs:训练的总epoch数
        log_dir:TensorBoard日志目录
        model_dir:模型保存目录
        idx_to_char:索引到字符的映射
        char_to_idx:字符到索引的映射
        generate_every:每隔多少个epoch生成示例
        save_every:每隔多少个epoch保存模型
        seed_texts:用于生成示例的种子文本
        print_gradients:是否打印梯度
    """

    writer = SummaryWriter(log_dir)
    
    # 如果没有提供种子文本，则使用默认种子
    if seed_texts is None:
        seed_texts = ["I love you", "When I was", "All I want"]
    

    best_val_loss = float('inf')
    

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_tokens = 0

        all_gradients = {}

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 调整输出和目标的形状以计算损失
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)

            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item() * targets.numel()
            total_train_tokens += targets.numel()

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), step)
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
       
            if print_gradients and batch_idx == 0:  # 只收集第一个批次的梯度
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        all_gradients[name] = grad_norm
        
        # 计算整体训练损失和困惑度
        avg_train_loss = total_train_loss / total_train_tokens
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        
        # 验证
        model.eval()
        total_val_loss = 0
        total_val_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                total_val_loss += loss.item() * targets.numel()
                total_val_tokens += targets.numel()

        avg_val_loss = total_val_loss / total_val_tokens
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/val', val_perplexity, epoch)

        if print_gradients:
            for name, grad_norm in all_gradients.items():
                writer.add_scalar(f'Gradients/{name}', grad_norm, epoch)
        
        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Train PPL: {train_perplexity:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val PPL: {val_perplexity:.4f}')
        
        # 保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, model_path)
            print(f'Best model saved at epoch {epoch}')

        if (epoch + 1) % save_every == 0:
            model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, model_path)
            print(f'Model saved at epoch {epoch+1}')
        
        # 定期生成示例文本
        if (epoch + 1) % generate_every == 0:
            for seed_text in seed_texts:
                generated_text = generate_text(
                    model=model,
                    char_to_idx=char_to_idx,
                    idx_to_char=idx_to_char,
                    seed_text=seed_text,
                    max_length=200,
                    temperature=0.8,
                    device=device
                )
                print(f'\nSeed: "{seed_text}"\nGenerated: "{generated_text}"\n')
                writer.add_text(f'Generated_Text/{seed_text}', generated_text, epoch)

    writer.close()


def train_and_evaluate(args):
    """训练和评估模型的主函数
    
    Args:
        args
    """
    # 设置随机种子
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 创建保存的目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, args.model_type, current_time)
    model_dir = os.path.join(args.model_dir, args.model_type, current_time)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print('Loading and preprocessing data...')
    lyrics_text = load_lyrics_data(args.data_path)
    preprocessed_text = preprocess_text(lyrics_text)
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(preprocessed_text)

    with open(os.path.join(model_dir, 'char_mappings.json'), 'w') as f:
        json.dump({
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': vocab_size
        }, f)

    train_text, val_text, test_text = split_data(preprocessed_text)
    train_loader, val_loader, test_loader = get_data_loaders(
        train_text, val_text, test_text, char_to_idx, 
        sequence_length=args.sequence_length, batch_size=args.batch_size
    )

    print(f'Creating {args.model_type} model...')
    if args.model_type == 'rnn':
        model = CharRNN(
            input_size=vocab_size,
            hidden_size=args.hidden_size,
            output_size=vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            sequence_length=args.sequence_length
        )
    elif args.model_type == 'lstm':
        model = CharLSTM(
            input_size=vocab_size,
            hidden_size=args.hidden_size,
            output_size=vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            sequence_length=args.sequence_length
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    # 保存模型配置
    model_config = {
        'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional if args.model_type == 'lstm' else False,
        'sequence_length': args.sequence_length,
        'vocab_size': vocab_size
    }
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 种子文本
    seed_texts = ["I love you", "When I was", "All I want", "She said"]

    print('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        log_dir=log_dir,
        model_dir=model_dir,
        idx_to_char=idx_to_char,
        char_to_idx=char_to_idx,
        generate_every=args.generate_every,
        save_every=args.save_every,
        seed_texts=seed_texts
    )
    
    # 加载最佳模型进行评估
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Evaluating on test set...')
    test_perplexity = calculate_perplexity(model, test_loader, device)
    print(f'Test Perplexity: {test_perplexity:.4f}')
    print('Generating text for evaluation...')
    all_generated_text = ""
    for seed_text in seed_texts:
        generated_text = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=seed_text,
            max_length=1000,
            temperature=0.8,
            device=device
        )
        all_generated_text += generated_text + "\n\n"
        print(f'\nSeed: "{seed_text}"\nGenerated: "{generated_text[:200]}..."\n')
    
    # 计算生成文本的多样性
    diversity_metrics = calculate_diversity(all_generated_text)
    print(f'Diversity Metrics: {diversity_metrics}')

    eval_results = {
        'test_perplexity': test_perplexity,
        'diversity_metrics': diversity_metrics
    }
    with open(os.path.join(model_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f)
    
    # 可视化字符分布
    print('Visualizing character distributions...')
    original_dist = calculate_char_distribution(preprocessed_text)
    generated_dist = calculate_char_distribution(all_generated_text)
    
    # 计算JS散度
    js_divergence = calculate_js_divergence(original_dist, generated_dist)
    print(f'Jensen-Shannon Divergence: {js_divergence:.4f}')
    js_div = {
        'js_divergence': js_divergence,
        'original_distribution': original_dist,
        'generated_distribution': generated_dist
    }
    with open(os.path.join(model_dir, 'eval_results.json'), 'w') as f:
        json.dump(js_div, f)
    
    # 分布对比图
    fig = plot_distribution_comparison(original_dist, generated_dist)
    plt.savefig(os.path.join(model_dir, 'char_distribution.png'))
    plt.close(fig)
    
    print(f'All results saved to {model_dir}')
    return model_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train and evaluate character-level language models')
    parser.add_argument('--data_path', type=str, default='../Corpus-of-Taylor-Swift-main/lyrics/flat-song-lyrics.json',
                        help='Path to the lyrics data file')
    parser.add_argument('--model_type', type=str, choices=['rnn', 'lstm'], default='lstm',
                        help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of the hidden layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Whether to use bidirectional LSTM (only for LSTM model)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Length of input sequences')
    parser.add_argument('--generate_every', type=int, default=10,
                        help='Generate sample text every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_dir', type=str, default='/root/Taylor/taylor_lyrics_generator/logs',
                        help='Directory to store TensorBoard logs')
    parser.add_argument('--model_dir', type=str, default='/root/Taylor/taylor_lyrics_generator/checkpoints',
                        help='Directory to store trained models')
    
    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == '__main__':
    main()
