#!/usr/bin/env python3
"""
激活PCA可视化脚本
将有害和无害数据的激活通过PCA降维到二维平面进行可视化
同时计算每对有害和无害样本之间的向量距离

对于DPO数据：计算配对的距离（每个rejected对应一个chosen）
对于PolyRefuse数据：如果样本数量相等则计算配对距离，否则计算所有组合的距离

注意：本脚本使用 output_hidden_states=True 方式提取每一层的输出（而非输入），
与 precompute_english_distances.py 保持一致。

使用方法：
1. DPO数据集：
   python activation_pca_visualization.py --data_type dpo --data_path /path/to/dpo/data.json

2. PolyRefuse数据集：
   python activation_pca_visualization.py --data_type polyrefuse --harmful_path /path/to/harmful.json --harmless_path /path/to/harmless.json
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
from typing import List, Tuple, Dict, Any
import argparse

from pipeline.model_utils.model_factory import construct_model_base


class ActivationExtractor:
    """激活提取器类"""

    def __init__(self, model_base):
        self.model_base = model_base
        self.n_layers = model_base.model.config.num_hidden_layers
        self.d_model = model_base.model.config.hidden_size

    def extract_activations(self, texts: List[str], positions=[-1], batch_size: int = 8) -> torch.Tensor:
        """
        提取激活（使用output_hidden_states方式，提取每一层的输出）

        Args:
            texts: 输入文本列表
            positions: 提取的位置，[-1]表示最后一个token，"mean"表示平均所有token
            batch_size: 批处理大小

        Returns:
            activations: [n_samples, n_layers, d_model]
        """
        n_samples = len(texts)
        device = self.model_base.model.device

        # 初始化结果张量
        activations = torch.zeros((n_samples, self.n_layers, self.d_model), dtype=torch.float32)

        # 处理批次
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations"):
            batch_texts = texts[i:i+batch_size]
            batch_size_actual = len(batch_texts)

            # Tokenize输入
            inputs = self.model_base.tokenize_instructions_fn(instructions=batch_texts)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # 前向传播，获取所有层的hidden states
            with torch.no_grad():
                outputs = self.model_base.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,  # 获取所有层的输出
                    use_cache=False
                )
            
            # outputs.hidden_states: tuple of (batch_size, seq_len, hidden_size)
            # 包含 embedding 输出 + 每一层的输出，共 n_layers + 1 个
            hidden_states = outputs.hidden_states
            
            # 对每一层提取指定位置的激活
            for layer in range(self.n_layers):
                # hidden_states[0] 是 embedding 输出
                # hidden_states[1] 是 layer 0 的输出
                # hidden_states[layer+1] 是 layer 的输出
                layer_hidden = hidden_states[layer + 1]  # [batch_size, seq_len, d_model]
                
                if positions == "mean":
                    # 平均所有有效token的激活（排除padding）
                    # 使用 attention_mask 来屏蔽 padding tokens
                    # attention_mask: [batch_size, seq_len], 1 表示有效 token，0 表示 padding
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(layer_hidden)  # [batch_size, seq_len, d_model]
                    masked_hidden = layer_hidden * mask_expanded  # 将 padding 位置置零
                    sum_hidden = masked_hidden.sum(dim=1)  # [batch_size, d_model]
                    valid_token_counts = attention_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
                    batch_result = sum_hidden / valid_token_counts  # [batch_size, d_model]
                else:
                    # 使用指定位置的激活
                    # 找到每个样本的最后一个有效token位置
                    last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
                    
                    if positions == [-1]:
                        # 提取最后一个有效token的hidden state
                        batch_result = layer_hidden[
                            torch.arange(batch_size_actual, device=device), 
                            last_token_indices
                        ]  # [batch_size, d_model]
                    else:
                        # 提取指定位置的激活（假设用户指定的位置都是有效的）
                        # 注意：这里假设用户提供的 positions 不会超出有效 token 范围
                        batch_result = layer_hidden[:, positions, :]  # [batch_size, len(positions), d_model]
                        if len(positions) == 1:
                            batch_result = batch_result.squeeze(1)  # [batch_size, d_model]
                
                # 将结果复制到主张量
                activations[i:i+batch_size_actual, layer, :] = batch_result.cpu()

        return activations


def load_dpo_data(data_path: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """
    加载DPO数据并处理成有害和无害文本

    Args:
        data_path: 数据文件路径
        max_samples: 最大样本数，None表示全部

    Returns:
        harmless_texts: 无害文本列表
        harmful_texts: 有害文本列表
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    harmless_texts = []
    harmful_texts = []

    for item in data:
        # 构建对话上下文
        conversation_text = ""
        for conv in item["conversations"]:
            conversation_text += conv["value"] + " "

        # 无害文本：对话 + 选择的回复
        harmless_text = conversation_text + item["chosen"]["value"]
        # harmless_text = item["chosen"]["value"]
        harmless_texts.append(harmless_text)

        # 有害文本：对话 + 拒绝的回复
        harmful_text = conversation_text + item["rejected"]["value"]
        # harmful_text = item["rejected"]["value"]

        harmful_texts.append(harmful_text)

    print(f"Loaded {len(harmless_texts)} harmless and {len(harmful_texts)} harmful samples")
    return harmless_texts, harmful_texts


def load_polyrefuse_data(harmful_path: str, harmless_path: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """
    加载PolyRefuse数据集

    Args:
        harmful_path: 有害数据文件路径
        harmless_path: 无害数据文件路径
        max_samples: 最大样本数，None表示全部

    Returns:
        harmless_texts: 无害文本列表
        harmful_texts: 有害文本列表
    """
    print(f"Loading PolyRefuse data from {harmful_path} and {harmless_path}...")

    # 加载有害数据
    with open(harmful_path, 'r', encoding='utf-8') as f:
        harmful_data = json.load(f)

    # 加载无害数据
    with open(harmless_path, 'r', encoding='utf-8') as f:
        harmless_data = json.load(f)

    if max_samples:
        harmful_data = harmful_data[:max_samples]
        harmless_data = harmless_data[:max_samples]

    harmful_texts = [item.get("instruction_translated", item.get("instruction")) for item in harmful_data]
    harmless_texts = [item.get("instruction_translated", item.get("instruction")) for item in harmless_data]

    print(f"Loaded {len(harmless_texts)} harmless and {len(harmful_texts)} harmful samples")
    return harmless_texts, harmful_texts


def apply_pca_and_plot(activations: torch.Tensor, labels: List[str], layer: int, position_type: str, save_dir: str):
    """
    对激活进行PCA降维并绘图

    Args:
        activations: [n_samples, d_model] 的激活张量
        labels: 样本标签列表
        layer: 层号
        position_type: 位置类型 ("last" 或 "mean")
        save_dir: 保存目录
    """
    # 转换为numpy数组
    activations_np = activations.numpy()

    # PCA降维到2D
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations_np)

    # 分离有害和无害数据
    harmful_indices = [i for i, label in enumerate(labels) if label == 'harmful']
    harmless_indices = [i for i, label in enumerate(labels) if label == 'harmless']

    harmful_points = activations_2d[harmful_indices]
    harmless_points = activations_2d[harmless_indices]

    # 绘图
    plt.figure(figsize=(10, 8))

    # 绘制有害数据点
    plt.scatter(harmful_points[:, 0], harmful_points[:, 1],
               c='red', alpha=0.6, label='Harmful', s=50, edgecolors='darkred')

    # 绘制无害数据点
    plt.scatter(harmless_points[:, 0], harmless_points[:, 1],
               c='blue', alpha=0.6, label='Harmless', s=50, edgecolors='darkblue')

    # 添加标签和标题
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Layer {layer} - {position_type.capitalize()} Token Activations (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加方差解释比例
    explained_var = pca.explained_variance_ratio_
    plt.text(0.02, 0.98, '.2f',
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    filename = f'layer_{layer:02d}_{position_type}_token_pca.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {filename}")


def calculate_and_save_distances(activations: torch.Tensor, labels: List[str], layer: int, position_type: str, save_dir: str, paired: bool = True):
    """
    计算有害样本和无害样本之间的向量距离并保存

    Args:
        activations: [n_samples, d_model] 的激活张量
        labels: 样本标签列表
        layer: 层号
        position_type: 位置类型 ("last" 或 "mean")
        save_dir: 保存目录
        paired: 是否计算配对距离（True）还是所有组合（False）
    """
    # 分离有害和无害数据
    harmful_indices = [i for i, label in enumerate(labels) if label == 'harmful']
    harmless_indices = [i for i, label in enumerate(labels) if label == 'harmless']

    harmful_activations = activations[harmful_indices]  # [n_harmful, d_model]
    harmless_activations = activations[harmless_indices]  # [n_harmless, d_model]

    print(f"Layer {layer}, {position_type} token: {len(harmful_indices)} harmful, {len(harmless_indices)} harmless samples")

    # 计算距离
    distances = []
    distance_info = []

    if paired and len(harmful_indices) == len(harmless_indices):
        # 计算配对距离（DPO数据的情况）
        for i in range(len(harmful_indices)):
            harmful_act = harmful_activations[i]
            harmless_act = harmless_activations[i]

            # 计算L2距离
            distance = torch.norm(harmful_act - harmless_act).item()
            distances.append(distance)
            distance_info.append({
                'pair_idx': i,
                'harmful_idx': harmful_indices[i],
                'harmless_idx': harmless_indices[i],
                'distance': distance
            })
    else:
        # 计算所有组合的距离
        for i, harmful_act in enumerate(harmful_activations):
            for j, harmless_act in enumerate(harmless_activations):
                # 计算L2距离
                distance = torch.norm(harmful_act - harmless_act).item()
                distances.append(distance)
                distance_info.append({
                    'harmful_idx': harmful_indices[i],
                    'harmless_idx': harmless_indices[j],
                    'distance': distance
                })

    # 保存距离信息
    os.makedirs(save_dir, exist_ok=True)
    filename = f'layer_{layer:02d}_{position_type}_token_distances.json'

    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
        json.dump({
            'layer': layer,
            'position_type': position_type,
            'n_harmful': len(harmful_indices),
            'n_harmless': len(harmless_indices),
            'total_pairs': len(distances),
            'paired': paired and len(harmful_indices) == len(harmless_indices),
            'distances_stats': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances))
            },
            'distance_pairs': distance_info
        }, f, indent=2)

    print(f"Saved distances: {filename} (paired: {paired and len(harmful_indices) == len(harmless_indices)}, mean: {np.mean(distances):.4f}, std: {np.std(distances):.4f})")


def main():
    parser = argparse.ArgumentParser(description='激活PCA可视化')
    parser.add_argument('--data_type', type=str, choices=['dpo', 'polyrefuse'], default='dpo',
                       help='数据类型：dpo 或 polyrefuse')
    parser.add_argument('--data_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/hanzhuowen02/refuse/MPO/data/dpo_en_demo.json',
                       help='DPO数据文件路径（当data_type=dpo时使用）')
                    
    parser.add_argument('--harmful_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/hanzhuowen02/refuse/Multilingual-Refusal/PolyRefuse/harmful_test_translated_en.json',
                       help='有害数据文件路径（当data_type=polyrefuse时使用）')
    parser.add_argument('--harmless_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/hanzhuowen02/refuse/Multilingual-Refusal/PolyRefuse/harmless_test_translated_en.json',
                       help='无害数据文件路径（当data_type=polyrefuse时使用）')
    
    parser.add_argument('--model_path', type=str,
                       default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/hanzhuowen02/models/huggingface.co/meta-llama/Llama-3.1-8B-Instruct',
                       help='模型路径')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='最大样本数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录（默认根据数据类型自动设置）')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model_base = construct_model_base(args.model_path, 'en')

    # 创建激活提取器
    extractor = ActivationExtractor(model_base)

    # 设置默认保存目录
    if args.save_dir is None:
        args.save_dir = f'activation_pca_plots_{args.data_type}'

    # 根据数据类型加载数据
    if args.data_type == 'dpo':
        harmless_texts, harmful_texts = load_dpo_data(args.data_path, args.max_samples)
        use_paired_distances = True  # DPO数据是配对的
    elif args.data_type == 'polyrefuse':
        harmless_texts, harmful_texts = load_polyrefuse_data(args.harmful_path, args.harmless_path, args.max_samples)
        use_paired_distances = len(harmful_texts) == len(harmless_texts)  # PolyRefuse数据可能不是配对的
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    print(f"Using paired distances: {use_paired_distances}")

    # 合并数据和标签
    all_texts = harmless_texts + harmful_texts
    all_labels = ['harmless'] * len(harmless_texts) + ['harmful'] * len(harmful_texts)

    # 提取激活（最后token）
    print("Extracting activations for last token...")
    activations_last = extractor.extract_activations(all_texts, positions=[-1], batch_size=args.batch_size)
    # activations_last: [n_samples, n_layers, d_model]

    # 提取激活（平均所有token）
    print("Extracting activations for mean of all tokens...")
    activations_mean = extractor.extract_activations(all_texts, positions="mean", batch_size=args.batch_size)
    # activations_mean: [n_samples, n_layers, d_model]

    # 为每一层生成PCA图和距离计算
    print("Generating PCA plots and calculating distances...")
    for layer in tqdm(range(model_base.model.config.num_hidden_layers), desc="Processing layers"):
        # 最后token的激活
        layer_activations_last = activations_last[:, layer, :]  # [n_samples, d_model]
        apply_pca_and_plot(layer_activations_last, all_labels, layer, "last", args.save_dir)
        calculate_and_save_distances(layer_activations_last, all_labels, layer, "last", args.save_dir, use_paired_distances)

        # 平均token的激活
        layer_activations_mean = activations_mean[:, layer, :]  # [n_samples, d_model]
        apply_pca_and_plot(layer_activations_mean, all_labels, layer, "mean", args.save_dir)
        calculate_and_save_distances(layer_activations_mean, all_labels, layer, "mean", args.save_dir, use_paired_distances)

    print(f"All plots saved to {args.save_dir}")


if __name__ == "__main__":
    main()
