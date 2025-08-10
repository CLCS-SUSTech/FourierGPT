import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
from tqdm import tqdm
import time

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 训练设备
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        
        for batch_idx, (sequences, lengths, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(sequences, lengths)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
            predictions: 预测结果
            true_labels: 真实标签
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for sequences, lengths, labels in tqdm(val_loader, desc="验证中"):
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(sequences, lengths)
                loss = self.criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 保存预测结果
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, predictions, true_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
              weight_decay=1e-5, patience=10, save_path='checkpoints'):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_path: 模型保存路径
            
        Returns:
            best_model_path: 最佳模型路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 优化器和调度器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None
        
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存模型
                model_path = os.path.join(save_path, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, model_path)
                best_model_path = model_path
                print(f"保存最佳模型: {model_path}")
            else:
                patience_counter += 1
                print(f"验证损失未改善，耐心计数: {patience_counter}/{patience}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                break
        
        return best_model_path
    
    def evaluate(self, test_loader, model_path=None):
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            model_path: 模型路径，如果为None则使用当前模型
            
        Returns:
            metrics: 评估指标字典
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型: {model_path}")
        
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for sequences, lengths, labels in tqdm(test_loader, desc="评估中"):
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences, lengths)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        auc = roc_auc_score(true_labels, probabilities)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }
        
        return metrics
    
    def plot_training_history(self, save_path='plots'):
        """Plot training history"""
        os.makedirs(save_path, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path='plots'):
        """Plot confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'ChatGPT'], 
                   yticklabels=['Human', 'ChatGPT'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, metrics, save_path='results'):
        """保存结果"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存指标
        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc': float(metrics['auc'])
        }
        
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存预测结果
        np.save(os.path.join(save_path, 'predictions.npy'), np.array(metrics['predictions']))
        np.save(os.path.join(save_path, 'true_labels.npy'), np.array(metrics['true_labels']))
        np.save(os.path.join(save_path, 'probabilities.npy'), np.array(metrics['probabilities']))
        
        print(f"结果已保存到: {save_path}")

if __name__ == "__main__":
    # 测试训练器
    from model import BiLSTMClassifier
    from data_loader import create_data_loaders
    
    # 创建模型
    model = BiLSTMClassifier(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True
    )
    
    # 创建训练器
    trainer = Trainer(model)
    
    print("训练器测试完成")
