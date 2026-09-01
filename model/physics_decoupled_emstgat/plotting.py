"""Plotting utilities for physics-decoupled EMSTGAT experiments."""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict, save_path: str = None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['loss'], label='Train Loss', color='#1f77b4')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', color='#ff7f0e')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['accuracy'], label='Train Acc', color='#1f77b4')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Acc', color='#ff7f0e')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: List[str] = None, save_path: str = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存: {save_path}")
    plt.close()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, 
                           top_n: int = 15, save_path: str = None):
    """绘制特征重要性条形图"""
    # 排序，实际显示数量取 top_n 和特征总数的较小值
    actual_n = min(top_n, len(feature_names))
    sorted_idx = np.argsort(importances)[::-1][:actual_n]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_names)), sorted_importances, color='#2196F3')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {actual_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"特征重要性图已保存: {save_path}")
    plt.close()
