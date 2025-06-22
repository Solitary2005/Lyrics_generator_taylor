import torch
import torch.nn as nn
import math


class CharRNN(nn.Module):
    """char-level RNN"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5, sequence_length=100):
        """
        Args:
            input_size
            hidden_size
            output_size
            num_layers: RNN层数
            dropout: Dropout
            sequence_length
        """
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.rnn = nn.RNN(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, nonlinearity='tanh')
            elif 'weight_hh' in name:
                #隐藏层到隐藏层，使用正交初始化
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.fc = nn.Linear(hidden_size, output_size)
        # Xavier初始化输出层
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x:[batch_size, seq_length]
            
        Returns:
            Tensor:[batch_size, seq_length, vocab_size]
        """
        batch_size = x.size(0)
        embedded = self.embedding(x)  
        #[batch_size, seq_length, hidden_size]
        h0 = self.init_hidden(batch_size).to(x.device)
        # 前向传播 RNN
        rnn_out, hidden = self.rnn(embedded, h0)
        #[batch_size, seq_length, hidden_size]
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        #[batch_size, seq_length, vocab_size]
        
        return output
    
    def init_hidden(self, batch_size):
        """
        初始化隐藏状态
        
        Args:
            batch_size
            
        Returns:
            Tensor
        """
        # 正态分布
        std = 1.0 / math.sqrt(self.hidden_size)
        return torch.randn(self.num_layers, batch_size, self.hidden_size) * std
