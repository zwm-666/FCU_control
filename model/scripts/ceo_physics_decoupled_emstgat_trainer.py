# CEO-QAAdamW-EnhancedMSTGAT 训练器
# 用于燃料电池状态分类

import os
import sys
import time
import json
import math
import gc
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

# ================== GPU 配置（必须在导入 TensorFlow 前设置环境变量）==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TF 日志输出

import tensorflow as tf
try:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.layers import Dense, RepeatVector
    from tensorflow.keras import backend as K
    KERAS = tf.keras
except ModuleNotFoundError:
    import keras as KERAS
    from keras import layers, Model
    from keras.layers import Dense, RepeatVector
    from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from physics_decoupled_emstgat import (
    PHYSICAL_FEATURE_GROUPS,
    PHYSICAL_GROUP_ORDER,
    EnhancedMSTGAT,
    ModelConfig,
    build_physical_feature_group_indices,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_history,
    select_physics_aware_features,
)

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


RESULT_SUBDIRS = ("figures", "metrics", "models", "logs")


def build_result_layout(output_root: str = "results") -> Dict[str, str]:
    """Return the canonical result directory layout for this trainer."""
    root = os.fspath(output_root)
    layout = {"root": root}
    layout.update({name: os.path.join(root, name) for name in RESULT_SUBDIRS})
    return layout


def ensure_result_layout(output_root: str = "results") -> Dict[str, str]:
    """Create and return the canonical result directory layout."""
    layout = build_result_layout(output_root)
    for path in layout.values():
        os.makedirs(path, exist_ok=True)
    return layout




# ================== GPU 配置模块 ==================
def setup_gpu(use_mixed_precision: bool = True, use_xla: bool = True) -> Dict[str, Any]:
    """配置 GPU 加速（针对 Kaggle 环境优化）
    
    Args:
        use_mixed_precision: 是否启用混合精度训练（推荐对于 P100/T4/V100）
        use_xla: 是否启用 XLA JIT 编译
        
    Returns:
        GPU 配置信息字典
    """
    gpu_info = {
        'gpus_available': [],
        'mixed_precision': False,
        'xla_enabled': False,
        'memory_growth': False
    }
    
    # 检测可用 GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        logger.info(f"✅ 检测到 {len(gpus)} 个 GPU:")
        for i, gpu in enumerate(gpus):
            logger.info(f"   GPU {i}: {gpu.name}")
            gpu_info['gpus_available'].append(gpu.name)
            
            # 启用内存增长（避免一次性占满显存）
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                gpu_info['memory_growth'] = True
            except RuntimeError as e:
                logger.warning(f"   无法设置内存增长: {e}")
        
        # 启用混合精度训练（float16）
        if use_mixed_precision:
            try:
                policy = KERAS.mixed_precision.Policy('mixed_float16')
                KERAS.mixed_precision.set_global_policy(policy)
                gpu_info['mixed_precision'] = True
                logger.info(f"✅ 已启用混合精度训练 (mixed_float16)")
            except Exception as e:
                logger.warning(f"⚠️ 无法启用混合精度: {e}")
        
        # 启用 XLA JIT 编译
        if use_xla:
            try:
                tf.config.optimizer.set_jit(True)
                gpu_info['xla_enabled'] = True
                logger.info(f"✅ 已启用 XLA JIT 编译")
            except Exception as e:
                logger.warning(f"⚠️ 无法启用 XLA: {e}")
    else:
        logger.warning("⚠️ 未检测到 GPU，将使用 CPU 训练")
    
    return gpu_info


def get_gpu_memory_info() -> str:
    """获取 GPU 显存使用信息"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 尝试使用 nvidia-smi（Kaggle 环境可用）
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(',')
                return f"{used.strip()}MB / {total.strip()}MB"
    except Exception:
        pass
    return "N/A"


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    patience: int = 15
    seed: int = 42



# ================== 优化器：QAAdamW_Lite ==================
class WarmupCosineDecay(KERAS.optimizers.schedules.LearningRateSchedule):
    """带 Warmup 的余弦衰减学习率调度
    
    修复：添加边界保护，确保学习率不会异常衰减
    """
    
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.01):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = float(max(total_steps, 1))
        self.warmup_steps = float(max(warmup_steps, 1))
        self.min_lr = self.base_lr * min_lr_ratio  # 最小学习率，防止衰减到0
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup 阶段
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        
        # Cosine 衰减阶段
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        # 确保 progress 在 [0, 1] 范围内
        progress = tf.minimum(decay_step / tf.maximum(decay_steps, 1.0), 1.0)
        progress = tf.maximum(progress, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * progress))
        decayed_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        
        # 选择阶段
        lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        # 确保最小学习率
        lr = tf.maximum(lr, self.min_lr)
        return lr
    
    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'total_steps': int(self.total_steps),
            'warmup_steps': int(self.warmup_steps),
            'min_lr': self.min_lr,
        }


class QAAdamW_Lite(KERAS.optimizers.Optimizer):
    """轻量级 QAAdamW 优化器（AdaBelief + 权重衰减 + 噪声注入）
    
    注意：为兼容 CEO 超参数搜索中不同 trial 的模型形状变化，
    本优化器使用字典而非列表存储累积变量。
    """
    
    def __init__(self, learning_rate=1e-3, weight_decay=1e-4, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, initial_noise_factor=0.02, use_adabelief=True,
                 total_steps=10000, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self._weight_decay = weight_decay
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._initial_noise_factor = initial_noise_factor
        self._use_adabelief = use_adabelief
        self._total_steps = total_steps
        # 使用字典存储累积变量，键为变量引用ID
        self._m_dict = {}
        self._v_dict = {}
    
    def build(self, var_list):
        super().build(var_list)
        # 每次 build 都根据当前 var_list 创建/更新累积变量
        for var in var_list:
            var_key = id(var)
            if var_key not in self._m_dict:
                self._m_dict[var_key] = self.add_variable_from_reference(var, 'm')
                self._v_dict[var_key] = self.add_variable_from_reference(var, 'v')
    
    def update_step(self, gradient, variable, learning_rate):
        # 处理稀疏梯度 (IndexedSlices)
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        
        var_key = id(variable)
        
        # 动态获取或创建累积变量（处理运行时新增的变量）
        if var_key not in self._m_dict:
            self._m_dict[var_key] = self.add_variable_from_reference(variable, 'm')
            self._v_dict[var_key] = self.add_variable_from_reference(variable, 'v')
        
        m = self._m_dict[var_key]
        v = self._v_dict[var_key]
        
        lr = tf.cast(learning_rate, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)
        beta1 = tf.cast(self._beta_1, variable.dtype)
        beta2 = tf.cast(self._beta_2, variable.dtype)
        eps = tf.cast(self._epsilon, variable.dtype)
        
        # 添加噪声（逐渐衰减）
        noise_factor = self._initial_noise_factor * tf.maximum(1.0 - step / self._total_steps, 0.0)
        noise = tf.random.normal(tf.shape(gradient), stddev=noise_factor, dtype=gradient.dtype)
        gradient = gradient + noise
        
        # AdaBelief or Adam
        m.assign(beta1 * m + (1 - beta1) * gradient)
        if self._use_adabelief:
            v.assign(beta2 * v + (1 - beta2) * tf.square(gradient - m))
        else:
            v.assign(beta2 * v + (1 - beta2) * tf.square(gradient))
        
        m_hat = m / (1 - tf.pow(beta1, step))
        v_hat = v / (1 - tf.pow(beta2, step))
        
        # 更新变量
        update = lr * m_hat / (tf.sqrt(v_hat) + eps)
        
        # 权重衰减
        if self._weight_decay > 0:
            variable.assign_sub(lr * self._weight_decay * variable)
        
        variable.assign_sub(update)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'weight_decay': self._weight_decay,
            'beta_1': self._beta_1,
            'beta_2': self._beta_2,
            'epsilon': self._epsilon,
            'initial_noise_factor': self._initial_noise_factor,
            'use_adabelief': self._use_adabelief,
            'total_steps': self._total_steps,
        })
        return config


# ================== 数据预处理函数（与原版一致）==================
def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理数据框中的缺失值（滚动中位数填充）"""
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"检测到 {missing_before} 个缺失值")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].rolling(window=5, min_periods=1, center=True).median()
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        logger.info(f"缺失值处理完成，剩余 {df.isnull().sum().sum()} 个缺失值")
    return df


def _remove_outliers(df: pd.DataFrame, label_col: Optional[str] = None, threshold: float = 3.0) -> pd.DataFrame:
    """使用Z-score方法检测并处理异常值（双指数平滑替换）"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    outliers_count = 0
    alpha, beta = 0.3, 0.1  # 双指数平滑参数
    for col in numeric_cols:
        if label_col and col == label_col:
            continue
        z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
        mask = z > threshold
        cnt = int(mask.sum())
        if cnt > 0:
            outliers_count += cnt
            valid = df.loc[~mask, col]
            if len(valid) >= 2:
                level = valid.iloc[0]
                trend = valid.iloc[1] - valid.iloc[0]
                smoothed = [level]
                for i in range(1, len(valid)):
                    last_level = level
                    level = alpha * valid.iloc[i] + (1 - alpha) * (level + trend)
                    trend = beta * (level - last_level) + (1 - beta) * trend
                    smoothed.append(level)
                # 用最近平滑近邻替换异常值
                outlier_idx = df[mask].index
                for idx in outlier_idx:
                    nearest = valid.index[np.abs(valid.index - idx).argmin()]
                    s_idx = valid.index.get_loc(nearest)
                    df.loc[idx, col] = smoothed[min(s_idx, len(smoothed) - 1)]
            else:
                df.loc[mask, col] = df[col].median()
    if outliers_count > 0:
        logger.info(f"处理异常值 {outliers_count} 个")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """移除重复行"""
    dup = int(df.duplicated().sum())
    if dup > 0:
        logger.info(f"检测到 {dup} 行重复数据，已移除")
        df = df.drop_duplicates(keep='first')
    return df


# ================== 数据加载（与原版一致）==================
def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """加载和预处理数据（与原版enhanced_mstgat_with_ceo.py一致）"""
    logger.info(f"加载数据: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"原始数据形状: {df.shape}")
    
    # 移除不需要的列（与原版一致）
    cols_to_remove = ['State', 'state', 'tsec']
    removed_cols = []
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_cols.append(col)
    if removed_cols:
        logger.info(f"已移除列: {removed_cols}")
    
    # 分离特征和标签（在数据清洗之前确定label_col）
    label_col = df.columns[-1]
    logger.info(f"标签列: {label_col}")
    
    # 数据清洗（与原版一致：缺失值填充 -> 异常值处理 -> 去重）
    df = _handle_missing_values(df)
    df = _remove_outliers(df, label_col=label_col)
    df = _remove_duplicates(df)
    
    logger.info(f"清洗后数据形状: {df.shape}")
    
    # 分离特征和标签
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    feature_names = list(df.columns[:-1])
    
    # 标签编码
    if not np.issubdtype(y.dtype, np.integer):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    num_classes = len(np.unique(y))
    logger.info(f"特征数: {X.shape[1]}, 类别数: {num_classes}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 特征选择（基于梯度提升的重要性）
    logger.info("计算特征重要性...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
    gb.fit(X_train, y_train)
    importances = gb.feature_importances_
    
    # 分组感知选择：先按全局累计重要性筛选，再保证各物理分支至少保留特征
    selected_idx = select_physics_aware_features(
        feature_names,
        importances,
        cumulative_threshold=0.95,
        min_features_per_group=1,
    )
    
    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]
    selected_features = [feature_names[i] for i in selected_idx]
    
    logger.info(f"选择 {len(selected_idx)}/{len(feature_names)} 个特征（分组感知累计重要性≥95%）")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train.astype(np.int32), 'y_test': y_test.astype(np.int32),
        'num_classes': num_classes,
        'feature_names': selected_features,
        'feature_importances': importances[selected_idx],
        'scaler': scaler,
    }


def create_datasets(X_train, y_train, X_val, y_val, batch_size: int = 32):
    """创建 TensorFlow 数据集"""
    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train, tf.float32), tf.cast(y_train, tf.int32))
    ).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_val, tf.float32), tf.cast(y_val, tf.int32))
    ).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds




# ================== CEO 优化算法（内联实现）==================
def _ceo_edm(x0, y0, itermax: int):
    """Exponential Discrete Memristor (E-DM) 混沌映射
    
    支持标量或向量初值：
    - 若 x0/y0 为标量，返回形状为 (itermax,) 的序列
    - 若 x0/y0 为向量，返回形状为 (itermax, Dim) 的序列
    """
    k = 2.66
    xo = np.asarray(x0, dtype=np.float64)
    yo = np.asarray(y0, dtype=np.float64)
    x_seq = []
    y_seq = []
    for _ in range(itermax):
        xn = k * (np.exp(-np.cos(np.pi * yo)) - 1.0) * xo
        yn = yo + xo
        x_seq.append(xn)
        y_seq.append(yn)
        xo, yo = xn, yn
    return np.array(x_seq), np.array(y_seq)


def _ceo_bound_constraint(v, lu):
    """处理边界约束违规"""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    low_state = np.tile(lu[0], (v.shape[0], 1))
    up_state = np.tile(lu[1], (v.shape[0], 1))
    # 处理下界违规
    vio_low = v < low_state
    v[vio_low] = np.minimum(up_state[vio_low], 2 * low_state[vio_low] - v[vio_low])
    # 处理上界违规
    vio_upper = v > up_state
    v[vio_upper] = np.maximum(low_state[vio_upper], 2 * up_state[vio_upper] - v[vio_upper])
    return v


def _ceo_binomial_crossover(p, v, CR):
    """二项交叉操作"""
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    # 确保 p 和 v 形状一致
    if p.shape[0] == 1 and v.shape[0] > 1:
        p = np.tile(p, (v.shape[0], 1))
    N, dim = v.shape
    j_rand = np.floor(np.random.rand(N) * dim).astype(int)
    t = np.random.rand(N, dim) < CR
    t[np.arange(N), j_rand] = True
    return np.where(t, v, p)


def ceo_optimize(func, Np: int, Dim: int, Varmin, Varmax, N: int = 20, max_iter: int = 100):
    """CEO (Chaotic Evolution Optimization) 优化算法
    
    Args:
        func: 目标函数（最小化）
        Np: 种群大小（必须为偶数）
        Dim: 问题维度
        Varmin: 下界数组
        Varmax: 上界数组
        N: 混沌样本数（默认20）
        max_iter: 最大迭代次数
        
    Returns:
        (最佳解, 最佳适应度, 历史记录)
    """
    # 确保 Np 为偶数
    if Np % 2 != 0:
        Np = max(2, Np + 1)
    
    # 处理搜索范围
    Varmin = np.asarray(Varmin, dtype=np.float64).flatten()
    Varmax = np.asarray(Varmax, dtype=np.float64).flatten()
    lu = np.vstack([Varmin, Varmax])
    if lu.shape[1] == 1:
        lu = np.tile(lu, (1, Dim))
    
    # 定义混沌搜索域
    low_chacos = np.array([-0.5, -0.25])
    up_chacos = np.array([0.5, 0.25])
    
    # 初始化种群
    Population = lu[0] + np.random.rand(Np, Dim) * (lu[1] - lu[0])
    fit = np.array([func(ind) for ind in Population])
    fBest = float(np.min(fit))
    Best = Population[np.argmin(fit)].copy()
    
    history = [fBest]
    counter = 0
    
    for t in range(1, max_iter + 1):
        oldfBest = fBest
        rand_num = np.random.permutation(Np)
        
        ub = np.max(Population, axis=0)
        lb = np.min(Population, axis=0)
        range_pop = ub - lb
        range_pop = np.where(range_pop < 1e-10, 1e-10, range_pop)
        
        for i in range(0, Np, 2):
            if i + 1 >= Np:
                break
            index = rand_num[i:i + 2]
            xy = Population[index]  # (2, Dim)
            
            # 区间映射：从问题空间 -> 混沌空间
            xy_dot = ((xy - lb) / range_pop) * (up_chacos - low_chacos)[:, None] + low_chacos[:, None]
            
            # 生成混沌个体
            x_chaos, y_chaos = _ceo_edm(xy_dot[0], xy_dot[1], N)
            chaos_total = np.vstack([x_chaos, y_chaos])  # (2*N, Dim)
            
            for k in range(2):
                xy_chaos = chaos_total[k * N:(k + 1) * N]  # (N, Dim)
                
                # 映射回实际优化问题空间
                xy_chaos_dot = ((xy_chaos - low_chacos[k]) / (up_chacos[k] - low_chacos[k])) * range_pop + lb
                
                # 变异
                if np.random.rand() < 0.5:
                    xy_hat = xy[k] + np.random.rand(N, 1) * (xy_chaos_dot - xy[k])
                else:
                    xy_hat = Best + np.random.rand(N, 1) * (xy_chaos_dot - xy[k])
                
                # 交叉
                CR = np.random.rand()
                xy_trial = _ceo_binomial_crossover(xy[k], xy_hat, CR)
                
                # 边界约束
                xy_trial = _ceo_bound_constraint(xy_trial, lu)
                
                # 适应度评估
                fit_xy_trial = np.array([func(trial) for trial in xy_trial])
                best_trial_idx = np.argmin(fit_xy_trial)
                
                # 选择
                if fit_xy_trial[best_trial_idx] < fit[index[k]]:
                    Population[index[k]] = xy_trial[best_trial_idx]
                    fit[index[k]] = fit_xy_trial[best_trial_idx]
        
        # 更新最佳解
        fBest = float(np.min(fit))
        Best = Population[np.argmin(fit)].copy()
        
        # 终止条件
        if abs(oldfBest - fBest) < 1e-8:
            counter += 1
            if counter > 50:
                break
        else:
            counter = 0
        
        history.append(fBest)
    
    return Best, fBest, history


# ================== 最优参数保存和加载 ==================
def _get_best_params_path(output_dir: str = 'results') -> str:
    """获取最优参数保存路径"""
    layout = ensure_result_layout(output_dir)
    return os.path.join(layout['metrics'], 'ceo_best_params.json')


def save_best_params(params: Dict[str, Any], val_accuracy: float, output_dir: str = 'results'):
    """保存最优超参数到JSON文件
    
    Args:
        params: 超参数字典
        val_accuracy: 验证准确率
        output_dir: 输出目录
    """
    filepath = _get_best_params_path(output_dir)
    data = {
        'val_accuracy': float(val_accuracy),
        'hyperparameters': params,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 最优参数已保存: {filepath} (val_acc={val_accuracy:.4f})")
    except Exception as e:
        logger.warning(f"⚠️ 保存最优参数失败: {e}")


def load_best_params(output_dir: str = 'results') -> Optional[Dict[str, Any]]:
    """加载历史最优超参数
    
    Returns:
        包含 'hyperparameters' 和 'val_accuracy' 的字典，如果文件不存在则返回 None
    """
    filepath = _get_best_params_path(output_dir)
    if not os.path.exists(filepath):
        logger.info(f"未找到历史最优参数文件: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ 已加载历史最优参数 (val_acc={data.get('val_accuracy', 0):.4f})")
        return data
    except Exception as e:
        logger.warning(f"⚠️ 加载最优参数失败: {e}")
        return None


# ================== CEO 超参数搜索 ==================
def define_search_space():
    """定义超参数搜索空间（基于最优参数配置）
    
    注意: 形状相关参数固定为最优值，避免与自定义优化器冲突。
    其他参数以最优值为中心进行小范围搜索。
    """
    return {
        # 可调参数（以最优值为中心的小范围搜索）
        'learning_rate': {'type': 'continuous', 'range': (5e-4, 1.5e-3)},  # 最优: 0.000813
        'weight_decay': {'type': 'continuous', 'range': (2e-4, 6e-4)},  # 最优: 0.000427
        'dropout_rate': {'type': 'continuous', 'range': (0.15, 0.25)},  # 最优: 0.206
        'initial_noise_factor': {'type': 'continuous', 'range': (0.02, 0.03)},  # 最优: 0.024
        
        # 固定参数（使用最优值）
        'hidden_units': {'type': 'discrete', 'choices': [192]},  # 最优: 192
        'attention_heads': {'type': 'discrete', 'choices': [16]},  # 最优: 16
        'max_sequence_length': {'type': 'discrete', 'choices': [15]},  # 最优: 15
        'knn_top_k': {'type': 'discrete', 'choices': [6]},  # 最优: 6
    }


def run_ceo_search(X_train, y_train, X_val, y_val, num_classes: int, 
                   budget: int = 10, epochs_per_trial: int = 25,
                   output_dir: str = 'results'):
    """运行 CEO 超参数搜索（使用内联的CEO算法）
    
    功能：
    1. 加载历史最优参数（如果存在）
    2. 运行CEO搜索找到新的最优参数
    3. 比较新旧参数，保留更好的
    4. 保存最终最优参数
    
    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        num_classes: 类别数
        budget: CEO搜索预算
        epochs_per_trial: 每次trial的训练轮数
        output_dir: 输出目录（用于保存最优参数）
        
    Returns:
        最优超参数字典
    """
    # ===== 1. 加载历史最优参数 =====
    historical_best = load_best_params(output_dir)
    historical_acc = 0.0
    if historical_best is not None:
        historical_acc = historical_best.get('val_accuracy', 0.0)
        logger.info(f"历史最优验证准确率: {historical_acc:.4f}")
    
    search_space = define_search_space()
    param_names = list(search_space.keys())
    dim = len(param_names)
    
    logger.info(f"开始 CEO 超参数搜索 (预算: {budget}, 维度: {dim})")
    
    def objective(x):
        """CEO目标函数 - 每次调用都创建全新的模型实例"""
        # ===== 1. 调用前先清理上一次的会话状态 =====
        KERAS.backend.clear_session()
        gc.collect()
        
        # ===== 2. 解析超参数 =====
        params = {}
        for i, name in enumerate(param_names):
            spec = search_space[name]
            if spec['type'] == 'continuous':
                lo, hi = spec['range']
                params[name] = float(lo + x[i] * (hi - lo))
            else:
                choices = spec['choices']
                idx = min(int(x[i] * len(choices)), len(choices) - 1)
                params[name] = choices[idx]
        
        # 确保 hidden_units 能被 attention_heads 整除
        params['hidden_units'] = (params['hidden_units'] // 8) * 8
        
        model = None
        val_acc = 0.0
        
        try:
            # ===== 3. 创建全新的模型配置（使用最优默认值）=====
            config = ModelConfig(
                input_shape=(X_train.shape[1],), 
                num_nodes=X_train.shape[1],
                num_classes=num_classes, 
                hidden_units=params['hidden_units'],
                attention_heads=params['attention_heads'],
                dropout_rate=params['dropout_rate'],
                max_sequence_length=params['max_sequence_length'],
                knn_top_k=params['knn_top_k']
                # dilation_rate, use_batch_norm, embedding_dim 使用 ModelConfig 默认值
            )
            
            # ===== 4. 创建全新的模型实例 =====
            model = EnhancedMSTGAT(config)
            
            # ===== 5. 创建优化器（CEO搜索使用标准Adam快速评估，与原版一致）=====
            optimizer = KERAS.optimizers.Adam(
                learning_rate=params['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            # ===== 6. 编译模型 =====
            model.compile(
                optimizer=optimizer,
                loss=KERAS.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )
            
            # ===== 7. 创建数据集 =====
            train_ds, val_ds = create_datasets(X_train, y_train, X_val, y_val)
            
            # ===== 8. 训练模型 =====
            history = model.fit(
                train_ds, 
                epochs=epochs_per_trial, 
                validation_data=val_ds, 
                verbose=0,
                callbacks=[
                    KERAS.callbacks.EarlyStopping(
                        monitor='val_accuracy', 
                        patience=5,  # 增加耐心
                        restore_best_weights=True
                    )
                ]
            )
            
            val_acc = float(max(history.history.get('val_accuracy', [0.0])))
            logger.info(f"  Trial: val_acc={val_acc:.4f}, params={params}")
            
        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            import traceback
            traceback.print_exc()
            val_acc = 0.0
        
        finally:
            # ===== 9. 彻底清理模型和会话 =====
            if model is not None:
                try:
                    del model
                except:
                    pass
            gc.collect()
            KERAS.backend.clear_session()
        
        return -val_acc
    
    # 运行 CEO（使用内联的ceo_optimize函数）
    Np = 4  # 种群大小（偶数）
    N = 20   # 混沌样本数
    max_iter = max(1, budget // 2)
    
    logger.info(f"CEO 参数: Np={Np}, N={N}, max_iter={max_iter}, dim={dim}")
    
    best_x, best_val, _ = ceo_optimize(
        func=objective, Np=Np, Dim=dim,
        Varmin=np.zeros(dim), Varmax=np.ones(dim),
        N=N, max_iter=max_iter
    )
    
    # 解码最佳参数
    best_params = {}
    for i, name in enumerate(param_names):
        spec = search_space[name]
        if spec['type'] == 'continuous':
            lo, hi = spec['range']
            best_params[name] = float(lo + best_x[i] * (hi - lo))
        else:
            choices = spec['choices']
            idx = min(int(best_x[i] * len(choices)), len(choices) - 1)
            best_params[name] = choices[idx]
    
    best_params['hidden_units'] = (best_params['hidden_units'] // 8) * 8
    
    # ===== 比较搜索结果与历史最优 =====
    search_val_acc = -best_val  # CEO返回的是负值（因为是最小化）
    logger.info(f"CEO 搜索完成！本次搜索最佳验证准确率: {search_val_acc:.4f}")
    
    # 比较并决定使用哪个参数
    if historical_best is not None and historical_acc > search_val_acc:
        # 历史参数更好，使用历史参数
        logger.info(f"⚠️ 本次搜索结果 ({search_val_acc:.4f}) 不如历史最优 ({historical_acc:.4f})")
        logger.info(f"✅ 使用历史最优参数")
        final_params = historical_best['hyperparameters']
        final_acc = historical_acc
    else:
        # 本次搜索更好或没有历史记录
        if historical_best is not None:
            logger.info(f"✅ 本次搜索结果 ({search_val_acc:.4f}) 优于历史最优 ({historical_acc:.4f})")
        else:
            logger.info(f"✅ 首次搜索，记录最优参数")
        final_params = best_params
        final_acc = search_val_acc
        # 保存新的最优参数
        save_best_params(final_params, final_acc, output_dir)
    
    logger.info(f"最终使用参数: {final_params}")
    
    return final_params


# ================== 主训练流程 ==================
def run_training_pipeline(csv_path: str, test_size: float = 0.2, epochs: int = 100,
                          budget: int = 10, seed: int = 42, use_gpu: bool = True,
                          skip_ceo: bool = False, output_dir: str = 'results'):
    """完整训练流程
    
    Args:
        csv_path: 数据集路径
        test_size: 测试集比例
        epochs: 训练轮数
        budget: CEO 搜索预算
        seed: 随机种子
        use_gpu: 是否使用 GPU 加速
        skip_ceo: 跳过 CEO 搜索，使用默认参数
        output_dir: 输出目录
    """
    start_time = time.time()
    
    # 创建输出目录
    result_layout = ensure_result_layout(output_dir)
    
    # GPU 配置
    if use_gpu:
        gpu_info = setup_gpu()
    else:
        gpu_info = {'gpus_available': [], 'mixed_precision': False}
    
    # 设置随机种子
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 加载数据
    data = load_and_preprocess_data(csv_path, test_size=test_size, seed=seed)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    num_classes = data['num_classes']
    feature_names = data['feature_names']
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )
    
    # CEO 超参数搜索
    if skip_ceo:
        # 使用最优超参数（来自 enhanced_mstgat_best_params_ts20.json）
        best_params = {
            'learning_rate': 0.0008127349453725082,
            'weight_decay': 0.00042655193657822774,
            'hidden_units': 192,
            'attention_heads': 16,  # 最优: 16
            'max_sequence_length': 15,  # 最优: 15
            'knn_top_k': 6,  # 最优: 6
            'dropout_rate': 0.20559374642540468,
            'initial_noise_factor': 0.024223792861225474,
            'dilation_rate': 4,  # 最优: 4
            'use_batch_norm': False,  # 最优: False
            'embedding_dim': 160  # 最优: 160
        }
        logger.info("使用最优超参数（跳过 CEO 搜索）")
    else:
        best_params = run_ceo_search(
            X_train, y_train, X_val, y_val, num_classes,
            budget=budget, epochs_per_trial=10, output_dir=output_dir
        )
        # CEO搜索完成后清理会话，确保最终模型在干净环境中创建
        KERAS.backend.clear_session()
        gc.collect()
    
    # 创建模型
    logger.info("\n创建最终模型...")
    feature_group_indices = build_physical_feature_group_indices(feature_names, X_train.shape[1])
    logger.info(f"物理分支特征划分: {feature_group_indices}")
    config = ModelConfig(
        input_shape=(X_train.shape[1],), 
        num_nodes=X_train.shape[1],
        num_classes=num_classes,
        hidden_units=best_params['hidden_units'],
        attention_heads=best_params['attention_heads'],
        dropout_rate=best_params['dropout_rate'],
        max_sequence_length=best_params['max_sequence_length'],
        knn_top_k=best_params['knn_top_k'],
        dilation_rate=best_params.get('dilation_rate', 4),
        use_batch_norm=best_params.get('use_batch_norm', False),
        embedding_dim=best_params.get('embedding_dim', 160),
        feature_group_indices=feature_group_indices,
    )
    
    model = EnhancedMSTGAT(config)
    model.build(input_shape=(None, X_train.shape[1]))
    logger.info(f"模型参数量: {model.count_params():,}")
    
    # 创建优化器
    steps_per_epoch = int(np.ceil(X_train.shape[0] / 32))
    total_steps = epochs * steps_per_epoch
    
    lr_schedule = WarmupCosineDecay(
        base_lr=best_params['learning_rate'],
        total_steps=total_steps,
        warmup_steps=int(0.1 * total_steps)
    )
    
    # 使用改进的 QAAdamW_Lite 优化器
    optimizer = QAAdamW_Lite(
        learning_rate=lr_schedule,
        weight_decay=best_params['weight_decay'],
        initial_noise_factor=best_params['initial_noise_factor'],
        use_adabelief=True,
        total_steps=total_steps
    )
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=KERAS.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 创建数据集
    train_ds, val_ds = create_datasets(X_train, y_train, X_val, y_val)
    
    # 回调函数
    # 注意：由于使用了自定义学习率调度器 WarmupCosineDecay，不再使用 ReduceLROnPlateau
    callbacks = [
        KERAS.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, mode='max', restore_best_weights=True
        ),
    ]
    
    # 训练
    logger.info("\n开始训练...")
    history = model.fit(
        train_ds, epochs=epochs, validation_data=val_ds,
        callbacks=callbacks, verbose=1
    )
    
    # 评估
    logger.info("\n评估模型...")
    pred_start = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    pred_time = time.time() - pred_start
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"测试结果:")
    logger.info(f"  准确率: {acc:.4f}")
    logger.info(f"  精确率: {prec:.4f}")
    logger.info(f"  召回率: {rec:.4f}")
    logger.info(f"  F1 分数: {f1:.4f}")
    logger.info(f"  预测时间: {pred_time:.4f}s")
    logger.info(f"{'='*60}")
    
    # 保存模型
    model_path = os.path.join(result_layout['models'], 'ceo_qaadam_emstgat.keras')
    model.save(model_path)
    logger.info(f"模型已保存: {model_path}")
    
    # 生成可视化
    logger.info("\n生成可视化...")
    plot_training_history(history.history, os.path.join(result_layout['figures'], 'training_history.png'))
    
    class_names = ['Normal', 'Light Flood', 'Heavy Flood', 'Dry'] if num_classes == 4 else None
    plot_confusion_matrix(y_test, y_pred, class_names, os.path.join(result_layout['figures'], 'confusion_matrix.png'))
    
    plot_feature_importance(feature_names, data['feature_importances'],
                           save_path=os.path.join(result_layout['figures'], 'feature_importance.png'))
    
    # 保存结果
    results = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
        'prediction_time': pred_time,
        'best_params': best_params,
        'gpu_info': gpu_info,
        'training_time': time.time() - start_time
    }
    
    with open(os.path.join(result_layout['metrics'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✅ 训练完成！总耗时: {time.time() - start_time:.2f}s")
    logger.info(f"   结果保存于: {result_layout['root']}")
    
    return results


def main():
    """CLI entry point."""
    # 数据集路径配置
    project_root = os.path.dirname(MODEL_DIR)  # pythonProject3
    
    # 数据集路径
    csv_path = os.path.join(project_root, '数据集2', 'fuel_cell_labeled_dataset_improved.csv')
    
    # 输出目录（统一保存在模型目录的 results 下）
    output_dir = os.path.join(MODEL_DIR, 'results')
    
    # 检查数据集是否存在
    if not os.path.exists(csv_path):
        logger.error(f"❌ 数据集文件不存在: {csv_path}")
        logger.info("请确保数据集文件位于正确的位置")
        sys.exit(1)
    
    logger.info(f"数据集路径: {csv_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 运行训练流程
    # skip_ceo=True: 跳过 CEO 超参数搜索，使用预设的最优参数（更快）
    # skip_ceo=False: 运行 CEO 超参数搜索（更慢，但可能找到更好的参数）
    results = run_training_pipeline(
        csv_path=csv_path,
        test_size=0.2,
        epochs=100,
        budget=10,
        seed=42,
        use_gpu=True,
        skip_ceo=True,  # 设置为 False 可启用 CEO 超参数搜索
        output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("训练完成！最终结果：")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1 分数: {results['f1_score']:.4f}")
    print("=" * 60)


# ================== 主入口 ==================
if __name__ == "__main__":
    main()
