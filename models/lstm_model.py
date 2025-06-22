import torch
import torch.nn as nn
import math


class CharLSTM(nn.Module):
    """char-level LSTM"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5, bidirectional=False, sequence_length=100):
        """
        Args:
            input_size
            hidden_size
            output_size
            num_layers:LSTM层数
            dropout
            bidirectional:是否使用双向LSTM
            sequence_length
        """
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.sequence_length = sequence_length
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 输入到隐藏状态使用Xavier
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # 隐藏状态到隐藏状态使用正交初始化
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.zeros_(param)
                # 遗忘门初始化为1
                param_data = param.data
                n = param_data.size(0)
                forget_gate_start = n // 4
                forget_gate_end = forget_gate_start + n // 4
                param_data[forget_gate_start:forget_gate_end].fill_(1.0)
        
        # 输出层，如果是双向LSTM，则输入维度需要乘以2
        output_dim = hidden_size * self.num_directions
        self.fc = nn.Linear(output_dim, output_size)
        # Xavier初始化输出层
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

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
        # [batch_size, seq_length, hidden_size]

        h0, c0 = self.init_hidden(batch_size)
        h0, c0 = h0.to(x.device), c0.to(x.device)

        lstm_out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        # [batch_size, seq_length, hidden_size * num_directions]

        lstm_out = self.dropout(lstm_out)

        output = self.fc(lstm_out)
        # [batch_size, seq_length, vocab_size]
        
        return output
    
    def init_hidden(self, batch_size):
        """
        初始化隐藏状态和细胞状态
        
        Args:
            batch_size
            
        Returns:
            tuple:(隐藏状态, 细胞状态)
        """
        # 正态分布初始化隐藏状态和细胞状态
        std = 1.0 / math.sqrt(self.hidden_size)
        hidden = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size) * std
        cell = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size) * std
        return hidden, cell
