import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.5, bidirectional=True):
        """
        初始化模型
        
        Args:
            input_size: 输入特征维度 (默认为1，因为每个时间步是一个标量)
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout: Dropout概率
            bidirectional: 是否使用双向LSTM
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_size = hidden_size * self.num_directions
        
        # MLP分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x, lengths=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            lengths: 序列长度 [batch_size]
            
        Returns:
            output: 分类输出 [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 如果输入是1D，扩展为2D [batch_size, seq_len] -> [batch_size, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # 打包序列（如果提供了长度信息）
        if lengths is not None:
            # 按长度排序（降序）
            sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
            _, original_indices = torch.sort(sorted_indices)
            
            x = x[sorted_indices]
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths.cpu(), batch_first=True
            )
        else:
            packed_x = x
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(packed_x)
        
        # 如果使用了pack_padded_sequence，需要解包
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            # 恢复原始顺序
            lstm_out = lstm_out[original_indices]
        
        # 获取最后一个时间步的输出
        # 对于双向LSTM，我们需要连接前向和后向的最后一个隐藏状态
        if self.bidirectional:
            # 获取前向和后向的最后一个隐藏状态
            forward_hidden = hidden[-2]  # 前向最后一层
            backward_hidden = hidden[-1]  # 后向最后一层
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # 恢复原始顺序（如果之前排序了）
        if lengths is not None:
            last_hidden = last_hidden[original_indices]
        
        # 通过分类器
        output = self.classifier(last_hidden)
        
        return output
    
    def get_attention_weights(self, x, lengths=None):
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入序列
            lengths: 序列长度
            
        Returns:
            attention_weights: 注意力权重
        """
        batch_size = x.size(0)
        
        # 如果输入是1D，扩展为2D
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM前向传播
        if lengths is not None:
            sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
            _, original_indices = torch.sort(sorted_indices)
            x = x[sorted_indices]
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths.cpu(), batch_first=True
            )
        else:
            packed_x = x
        
        lstm_out, _ = self.lstm(packed_x)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            lstm_out = lstm_out[original_indices]
        
        # 计算注意力权重（简单的点积注意力）
        # 使用最后一个隐藏状态作为查询
        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            query = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            query = hidden[-1]
        
        if lengths is not None:
            query = query[original_indices]
        
        # 计算注意力分数
        attention_scores = torch.bmm(lstm_out, query.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights

class SimpleLSTMClassifier(nn.Module):
    """简化版LSTM分类器（用于对比）"""
    
    def __init__(self, input_size=1, hidden_size=64, num_classes=2, dropout=0.3):
        super(SimpleLSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        lstm_out, (hidden, _) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        output = self.classifier(last_output)
        return output

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    seq_len = 50
    input_size = 1
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len)
    lengths = torch.tensor([seq_len, seq_len-10, seq_len-20, seq_len-30])
    
    # 测试BiLSTM模型
    model = BiLSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True
    )
    
    print(f"BiLSTM模型参数数量: {count_parameters(model):,}")
    
    # 前向传播测试
    output = model(x, lengths)
    print(f"输出形状: {output.shape}")
    print(f"输出: {output}")
    
    # 测试简化模型
    simple_model = SimpleLSTMClassifier(
        input_size=input_size,
        hidden_size=64,
        num_classes=2
    )
    
    print(f"\n简化模型参数数量: {count_parameters(simple_model):,}")
    
    simple_output = simple_model(x, lengths)
    print(f"简化模型输出形状: {simple_output.shape}")
