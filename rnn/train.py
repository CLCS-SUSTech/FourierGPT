#!/usr/bin/env python3
"""
RNN分类模型训练脚本

使用双向LSTM对负对数似然分数进行分类，区分人类和ChatGPT生成的文本。
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_loaders
from model import BiLSTMClassifier, SimpleLSTMClassifier, count_parameters
from trainer import Trainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RNN分类模型训练')
    
    # 数据参数
    parser.add_argument('--human_file', type=str, 
                       default='../data/poem_nll/poem_Human_llama3-8b-instruct.txt',
                       help='人类数据文件路径')
    parser.add_argument('--chatgpt_file', type=str,
                       default='../data/poem_nll/poem_ChatGPT_poem_llama3-8b-instruct.txt',
                       help='ChatGPT数据文件路径')
    parser.add_argument('--max_length', type=int, default=None,
                       help='序列最大长度，None表示使用所有数据')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='bilstm',
                       choices=['bilstm', 'simple'],
                       help='模型类型：bilstm或simple')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout概率')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='是否使用双向LSTM')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备，None表示自动选择')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='是否保存图表')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args):
    """创建模型"""
    if args.model_type == 'bilstm':
        model = BiLSTMClassifier(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:
        model = SimpleLSTMClassifier(
            input_size=1,
            hidden_size=args.hidden_size,
            num_classes=2,
            dropout=args.dropout
        )
    
    return model

def print_model_info(model, args):
    """打印模型信息"""
    print("=" * 60)
    print("模型配置")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"隐藏层大小: {args.hidden_size}")
    print(f"LSTM层数: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"双向LSTM: {args.bidirectional}")
    print(f"参数数量: {count_parameters(model):,}")
    print("=" * 60)

def print_data_info(train_loader, val_loader, test_loader):
    """打印数据信息"""
    print("=" * 60)
    print("数据信息")
    print("=" * 60)
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 计算样本数量
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    total_samples = train_samples + val_samples + test_samples
    
    print(f"训练样本数: {train_samples}")
    print(f"验证样本数: {val_samples}")
    print(f"测试样本数: {test_samples}")
    print(f"总样本数: {total_samples}")
    print("=" * 60)

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {args.device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        human_file=args.human_file,
        chatgpt_file=args.chatgpt_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=0
    )
    
    print_data_info(train_loader, val_loader, test_loader)
    
    # 创建模型
    print("创建模型...")
    model = create_model(args)
    print_model_info(model, args)
    
    # 创建训练器
    trainer = Trainer(model, device=args.device)
    
    # 训练模型
    print("开始训练...")
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    best_model_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=checkpoint_dir
    )
    
    # 评估模型
    print("评估模型...")
    results_dir = os.path.join(output_dir, 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    
    metrics = trainer.evaluate(test_loader, best_model_path)
    
    # 打印结果
    print("=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("=" * 60)
    
    # 保存结果
    trainer.save_results(metrics, results_dir)
    
    # 绘制图表
    if args.save_plots:
        print("绘制图表...")
        trainer.plot_training_history(plots_dir)
        trainer.plot_confusion_matrix(metrics['confusion_matrix'], plots_dir)
    
    # 保存配置
    config = {
        'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'max_length': args.max_length,
        'device': args.device,
        'seed': args.seed,
        'best_model_path': best_model_path,
        'results': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc': float(metrics['auc'])
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"训练完成！结果保存在: {output_dir}")
    print(f"最佳模型: {best_model_path}")

if __name__ == "__main__":
    main()
