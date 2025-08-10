#!/usr/bin/env python3
"""
测试RNN训练管道

这个脚本用于快速测试整个训练管道是否正常工作。
"""

import os
import sys
import torch
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_loaders
from model import BiLSTMClassifier, count_parameters
from trainer import Trainer

def test_data_loader():
    """测试数据加载器"""
    print("测试数据加载器...")
    
    human_file = "../data/poem_nll/poem_Human_llama3-8b-instruct.txt"
    chatgpt_file = "../data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt"
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            human_file=human_file,
            chatgpt_file=chatgpt_file,
            batch_size=8,
            max_length=50,  # 限制序列长度以加快测试
            shuffle=True,
            num_workers=0
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  训练集批次数: {len(train_loader)}")
        print(f"  验证集批次数: {len(val_loader)}")
        print(f"  测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        for sequences, lengths, labels in train_loader:
            print(f"  批次形状: {sequences.shape}")
            print(f"  序列长度: {lengths}")
            print(f"  标签: {labels}")
            break
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return None, None, None

def test_model():
    """测试模型"""
    print("\n测试模型...")
    
    try:
        model = BiLSTMClassifier(
            input_size=1,
            hidden_size=64,  # 使用较小的隐藏层以加快测试
            num_layers=2,
            num_classes=2,
            dropout=0.3,
            bidirectional=True
        )
        
        print(f"✓ 模型创建成功")
        print(f"  参数数量: {count_parameters(model):,}")
        
        # 测试前向传播
        batch_size = 4
        seq_len = 30
        x = torch.randn(batch_size, seq_len)
        lengths = torch.tensor([seq_len, seq_len-5, seq_len-10, seq_len-15])
        
        output = model(x, lengths)
        print(f"  输出形状: {output.shape}")
        print(f"  输出: {output}")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return None

def test_trainer():
    """测试训练器"""
    print("\n测试训练器...")
    
    try:
        # 创建小模型
        model = BiLSTMClassifier(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            num_classes=2,
            dropout=0.2,
            bidirectional=False
        )
        
        trainer = Trainer(model, device='cpu')  # 使用CPU以加快测试
        print(f"✓ 训练器创建成功")
        
        return trainer
        
    except Exception as e:
        print(f"✗ 训练器测试失败: {e}")
        return None

def test_full_pipeline():
    """测试完整管道"""
    print("\n测试完整管道...")
    
    try:
        # 创建数据加载器
        human_file = "../data/poem_nll/poem_Human_llama3-8b-instruct.txt"
        chatgpt_file = "../data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt"
        
        train_loader, val_loader, test_loader = create_data_loaders(
            human_file=human_file,
            chatgpt_file=chatgpt_file,
            batch_size=4,
            max_length=20,  # 使用很短的序列以加快测试
            shuffle=True,
            num_workers=0
        )
        
        # 创建模型
        model = BiLSTMClassifier(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            num_classes=2,
            dropout=0.1,
            bidirectional=False
        )
        
        # 创建训练器
        trainer = Trainer(model, device='cpu')
        
        # 运行一个epoch的训练
        print("运行一个训练epoch...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  训练准确率: {train_acc:.2f}%")
        
        # 运行验证
        print("运行验证...")
        val_loss, val_acc, _, _ = trainer.validate(val_loader)
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证准确率: {val_acc:.2f}%")
        
        # 运行评估
        print("运行评估...")
        metrics = trainer.evaluate(test_loader)
        print(f"  测试准确率: {metrics['accuracy']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        
        print("✓ 完整管道测试成功")
        return True
        
    except Exception as e:
        print(f"✗ 完整管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("RNN训练管道测试")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试各个组件
    success = True
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = test_data_loader()
    if train_loader is None:
        success = False
    
    # 测试模型
    model = test_model()
    if model is None:
        success = False
    
    # 测试训练器
    trainer = test_trainer()
    if trainer is None:
        success = False
    
    # 测试完整管道
    if success:
        pipeline_success = test_full_pipeline()
        if not pipeline_success:
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！训练管道可以正常使用。")
        print("\n要开始训练，请运行:")
        print("python train.py --batch_size 16 --num_epochs 10 --max_length 100")
    else:
        print("✗ 部分测试失败，请检查错误信息。")
    print("=" * 60)

if __name__ == "__main__":
    main()
