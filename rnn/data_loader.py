import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class NLLDataset(Dataset):
    """负对数似然分数数据集"""
    
    def __init__(self, human_file, chatgpt_file, max_length=None):
        """
        初始化数据集
        
        Args:
            human_file: 人类数据文件路径
            chatgpt_file: ChatGPT数据文件路径
            max_length: 序列最大长度，如果为None则使用所有数据
        """
        self.human_file = human_file
        self.chatgpt_file = chatgpt_file
        self.max_length = max_length
        
        # 加载数据
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """加载数据文件"""
        data = []
        labels = []
        
        # 加载人类数据 (label 0)
        with open(self.human_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    scores = [float(x) for x in line.split()]
                    if self.max_length:
                        scores = scores[:self.max_length]
                    data.append(scores)
                    labels.append(0)
        
        # 加载ChatGPT数据 (label 1)
        with open(self.chatgpt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    scores = [float(x) for x in line.split()]
                    if self.max_length:
                        scores = scores[:self.max_length]
                    data.append(scores)
                    labels.append(1)
        
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        scores = self.data[idx]
        label = self.labels[idx]
        
        # 转换为tensor
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return scores_tensor, label_tensor

def collate_fn(batch):
    """
    自定义批处理函数，处理变长序列
    
    Args:
        batch: 批次数据
        
    Returns:
        padded_sequences: 填充后的序列
        lengths: 原始序列长度
        labels: 标签
    """
    # 分离序列和标签
    sequences, labels = zip(*batch)
    
    # 获取每个序列的长度
    lengths = [len(seq) for seq in sequences]
    
    # 填充序列
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )
    
    # 转换为tensor
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, lengths, labels

def create_data_loaders(human_file, chatgpt_file, batch_size=32, 
                       train_split=0.8, val_split=0.1, max_length=None, 
                       shuffle=True, num_workers=0):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        human_file: 人类数据文件路径
        chatgpt_file: ChatGPT数据文件路径
        batch_size: 批次大小
        train_split: 训练集比例
        val_split: 验证集比例
        max_length: 序列最大长度
        shuffle: 是否打乱数据
        num_workers: 数据加载器工作进程数
        
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 创建完整数据集
    dataset = NLLDataset(human_file, chatgpt_file, max_length)
    
    # 计算分割点
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    human_file = "../data/poem_nll/poem_Human_llama3-8b-instruct.txt"
    chatgpt_file = "../data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        human_file, chatgpt_file, batch_size=8, max_length=100
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch_idx, (sequences, lengths, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  序列形状: {sequences.shape}")
        print(f"  长度: {lengths}")
        print(f"  标签: {labels}")
        break
