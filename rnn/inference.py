#!/usr/bin/env python3
"""
RNN模型推理脚本

用于加载训练好的模型并对新的负对数似然分数序列进行预测。
"""

import torch
import numpy as np
import argparse
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import BiLSTMClassifier, SimpleLSTMClassifier

class NLLPredictor:
    """负对数似然分数预测器"""
    
    def __init__(self, model_path, model_type='bilstm', device='cpu'):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            model_type: 模型类型 ('bilstm' 或 'simple')
            device: 推理设备
        """
        self.device = device
        self.model_type = model_type
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        """加载模型"""
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 根据模型类型创建模型
        if self.model_type == 'bilstm':
            model = BiLSTMClassifier(
                input_size=1,
                hidden_size=128,  # 匹配训练时的配置
                num_layers=2,
                num_classes=2,
                dropout=0.5,
                bidirectional=True
            )
        else:
            model = SimpleLSTMClassifier(
                input_size=1,
                hidden_size=64,
                num_classes=2,
                dropout=0.3
            )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict(self, scores, max_length=None):
        """
        预测单个序列
        
        Args:
            scores: 负对数似然分数列表或numpy数组
            max_length: 最大序列长度，None表示使用所有数据
            
        Returns:
            prediction: 预测标签 (0: 人类, 1: ChatGPT)
            probability: 预测概率
        """
        # 预处理数据
        if isinstance(scores, list):
            scores = np.array(scores)
        
        # 限制长度
        if max_length and len(scores) > max_length:
            scores = scores[:max_length]
        
        # 转换为tensor
        scores_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)  # 添加batch维度
        scores_tensor = scores_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(scores_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            probability = probabilities[0, prediction].item()
        
        return prediction, probability
    
    def predict_batch(self, scores_list, max_length=None):
        """
        批量预测
        
        Args:
            scores_list: 负对数似然分数列表的列表
            max_length: 最大序列长度
            
        Returns:
            predictions: 预测标签列表
            probabilities: 预测概率列表
        """
        predictions = []
        probabilities = []
        
        for scores in scores_list:
            pred, prob = self.predict(scores, max_length)
            predictions.append(pred)
            probabilities.append(prob)
        
        return predictions, probabilities

def load_scores_from_file(file_path, max_length=None):
    """从文件加载负对数似然分数"""
    scores_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                scores = [float(x) for x in line.split()]
                if max_length:
                    scores = scores[:max_length]
                scores_list.append(scores)
    
    return scores_list

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RNN模型推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='bilstm',
                       choices=['bilstm', 'simple'],
                       help='模型类型')
    parser.add_argument('--input_file', type=str,
                       help='输入文件路径（可选）')
    parser.add_argument('--scores', type=str,
                       help='负对数似然分数，用空格分隔（可选）')
    parser.add_argument('--max_length', type=int, default=None,
                       help='最大序列长度')
    parser.add_argument('--device', type=str, default='cpu',
                       help='推理设备')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = NLLPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    print(f"模型加载成功: {args.model_path}")
    print(f"模型类型: {args.model_type}")
    print(f"推理设备: {args.device}")
    print("-" * 50)
    
    if args.input_file:
        # 从文件加载数据
        print(f"从文件加载数据: {args.input_file}")
        scores_list = load_scores_from_file(args.input_file, args.max_length)
        
        print(f"加载了 {len(scores_list)} 个序列")
        
        # 批量预测
        predictions, probabilities = predictor.predict_batch(scores_list, args.max_length)
        
        # 输出结果
        print("\n预测结果:")
        print("序列ID\t预测\t概率\t\t类别")
        print("-" * 40)
        
        human_count = 0
        chatgpt_count = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = "人类" if pred == 0 else "ChatGPT"
            if pred == 0:
                human_count += 1
            else:
                chatgpt_count += 1
            
            print(f"{i+1}\t{pred}\t{prob:.4f}\t\t{label}")
        
        print("-" * 40)
        print(f"人类: {human_count} ({human_count/len(predictions)*100:.1f}%)")
        print(f"ChatGPT: {chatgpt_count} ({chatgpt_count/len(predictions)*100:.1f}%)")
        
    elif args.scores:
        # 从命令行参数加载数据
        scores = [float(x) for x in args.scores.split()]
        
        if args.max_length:
            scores = scores[:args.max_length]
        
        print(f"输入序列长度: {len(scores)}")
        
        # 预测
        prediction, probability = predictor.predict(scores, args.max_length)
        
        # 输出结果
        label = "人类" if prediction == 0 else "ChatGPT"
        print(f"\n预测结果:")
        print(f"预测标签: {prediction} ({label})")
        print(f"预测概率: {probability:.4f}")
        
    else:
        # 演示模式
        print("演示模式 - 使用示例数据")
        
        # 创建一些示例数据
        example_scores = [
            [11.13528, 3.72001, 10.88363, 10.93882, 0.16693, 0.18061, 5.67179, 2.59403, 0.70261, 2.07792],
            [9.04648, 5.01506, 2.77325, 0.00815, 8.66432, 9.81500, 4.27468, 1.97760, 3.49640, 5.45108],
            [10.76282, 9.75742, 6.83630, 8.46655, 5.38614, 1.13510, 0.98469, 2.61917, 0.22812, 3.53722]
        ]
        
        print(f"示例序列数量: {len(example_scores)}")
        
        # 批量预测
        predictions, probabilities = predictor.predict_batch(example_scores, args.max_length)
        
        # 输出结果
        print("\n预测结果:")
        print("序列ID\t预测\t概率\t\t类别")
        print("-" * 40)
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = "人类" if pred == 0 else "ChatGPT"
            print(f"{i+1}\t{pred}\t{prob:.4f}\t\t{label}")

if __name__ == "__main__":
    main()
