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


def calculate_and_save_distances(activations: torch.Tensor, labels: List[str], layer: int, position_type: str, save_dir: str, lang: str, paired: bool = True):
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
    filename = f'{lang}_layer_{layer:02d}_{position_type}_token_distances.json'

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


def plot_distance_hist_en_vs_other(
    args,
    layer: int,
    position_type: str,
    other_lang: str,
    bins: int = 50
):
    """
    在同一张图中对比 EN vs 另一语言 的距离分布（直方图）
    """

    en_fname = f'{args.save_dir}/en_layer_{layer:02d}_{position_type}_token_distances.json'
    other_fname = f'{args.save_dir}/{other_lang}_layer_{layer:02d}_{position_type}_token_distances.json'

    with open(en_fname, 'r', encoding='utf-8') as f:
        data_en = json.load(f)
    with open(other_fname, 'r', encoding='utf-8') as f:
        data_ot = json.load(f)

    dist_en = [d['distance'] for d in data_en['distance_pairs']]
    dist_ot = [d['distance'] for d in data_ot['distance_pairs']]

    plt.figure(figsize=(8, 6))

    plt.hist(
        dist_en, bins=bins, density=True,
        alpha=0.5, color='tab:blue', label='English'
    )
    plt.hist(
        dist_ot, bins=bins, density=True,
        alpha=0.5, color='tab:orange', label=other_lang
    )

    plt.xlabel('L2 Distance')
    plt.ylabel('Density')
    plt.title(
        f'Layer {layer} | {position_type.capitalize()} token\n'
        f'Distance Distribution: EN vs {other_lang}'
    )
    plt.legend()
    plt.grid(alpha=0.3)

    out_dir = os.path.join(args.save_dir, f'en_vs_{other_lang}')
    os.makedirs(out_dir, exist_ok=True)

    out_name = f'layer_{layer:02d}_{position_type}_distance_hist_en_vs_{other_lang}.png'
    plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_box_en_vs_other(
    args,
    layer: int,
    position_type: str,
    other_lang: str
):
    """
    EN vs 另一语言 的距离箱线图（更偏统计展示）
    """

    en_fname = f'{args.save_dir}/en_layer_{layer:02d}_{position_type}_token_distances.json'
    other_fname = f'{args.save_dir}/{other_lang}_layer_{layer:02d}_{position_type}_token_distances.json'

    with open(en_fname, 'r', encoding='utf-8') as f:
        data_en = json.load(f)
    with open(other_fname, 'r', encoding='utf-8') as f:
        data_ot = json.load(f)

    dist_en = [d['distance'] for d in data_en['distance_pairs']]
    dist_ot = [d['distance'] for d in data_ot['distance_pairs']]

    plt.figure(figsize=(6, 6))
    plt.boxplot(
        [dist_en, dist_ot],
        labels=['English', other_lang],
        showfliers=False
    )

    plt.ylabel('L2 Distance')
    plt.title(
        f'Layer {layer} | {position_type.capitalize()} token\n'
        f'Distance Comparison'
    )
    plt.grid(axis='y', alpha=0.3)

    out_dir = os.path.join(args.save_dir, f'en_vs_{other_lang}')
    os.makedirs(out_dir, exist_ok=True)

    out_name = f'layer_{layer:02d}_{position_type}_distance_box_en_vs_{other_lang}.png'
    plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_hist_all_langs(
    args,
    layer: int,
    position_type: str,
    langs = ('en', 'zh', 'ja', 'ko', 'ar', 'bn', 'sw'),
    bins: int = 100,
    figsize=(14, 4)  # 扁长
):
    """
    在同一张图中对比 7 种语言 的距离分布（直方图，density=True）。
    默认输出到: {save_dir}/all_langs/
    """
    dist_by_lang = {}

    # 读取每种语言的距离
    for lang in langs:
        fname = os.path.join(
            args.save_dir, f'{lang}_layer_{layer:02d}_{position_type}_token_distances.json'
        )
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dist_by_lang[lang] = np.array([d['distance'] for d in data['distance_pairs']], dtype=np.float32)

    # 统一 bins：用所有语言的全局范围，避免每个语言单独决定 bin 导致不可比
    all_dist = np.concatenate(list(dist_by_lang.values()), axis=0)
    dmin, dmax = float(all_dist.min()), float(all_dist.max())
    bin_edges = np.linspace(dmin, dmax, bins + 1)

    plt.figure(figsize=figsize)

    # 叠加绘制
    for lang in langs:
        plt.hist(
            dist_by_lang[lang],
            bins=bin_edges,
            density=True,
            alpha=0.8,
            label=lang
        )

    plt.xlabel('L2 Distance')
    plt.ylabel('Density')
    plt.title(f'Layer {layer} | {position_type.capitalize()} token | Distance Distribution (7 langs)')
    plt.legend(ncol=7, fontsize=9, frameon=False)
    plt.grid(alpha=0.25)

    out_dir = os.path.join(args.save_dir, 'all_langs')
    os.makedirs(out_dir, exist_ok=True)
    out_name = f'layer_{layer:02d}_{position_type}_distance_hist_all_langs.png'
    plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_box_all_langs(
    args,
    layer: int,
    position_type: str,
    langs = ('en', 'zh', 'ja', 'ko', 'ar', 'bn', 'sw'),
    figsize=(14, 4)  # 扁长
):
    """
    在同一张图中对比 7 种语言 的距离箱线图（showfliers=False）。
    默认输出到: {save_dir}/all_langs/
    """
    data_list = []
    for lang in langs:
        fname = os.path.join(
            args.save_dir, f'{lang}_layer_{layer:02d}_{position_type}_token_distances.json'
        )
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dist = [d['distance'] for d in data['distance_pairs']]
        data_list.append(dist)

    plt.figure(figsize=figsize)
    plt.boxplot(
        data_list,
        labels=list(langs),
        showfliers=False
    )
    plt.ylabel('L2 Distance')
    plt.title(f'Layer {layer} | {position_type.capitalize()} token | Distance Comparison (7 langs)')
    plt.grid(axis='y', alpha=0.25)

    out_dir = os.path.join(args.save_dir, 'all_langs')
    os.makedirs(out_dir, exist_ok=True)
    out_name = f'layer_{layer:02d}_{position_type}_distance_box_all_langs.png'
    plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='激活PCA可视化')
    parser.add_argument('--data_type', type=str, choices=['dpo', 'polyrefuse'], default='polyrefuse',
                       help='数据类型：dpo 或 polyrefuse')
    parser.add_argument('--data_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/hanzhuowen02/refuse/MPO/data/dpo_en_demo.json',
                       help='DPO数据文件路径（当data_type=dpo时使用）')
                    
    parser.add_argument('--harmful_dir', type=str, default="Multilingual-Refusal/dataset/splits_multi/",
                    help='有害数据路径（polyrefuse 时使用）')
    parser.add_argument('--harmless_dir', type=str, default="Multilingual-Refusal/dataset/splits_multi/",
                    help='无害数据路径（polyrefuse 时使用）')
    
    parser.add_argument('--model_path', type=str,
                       default='/ds/models/llms/Llama-3.1-8B-Instruct',
                       help='模型路径')
    parser.add_argument('--model_name', type=str,
                       default='llama',
                       help='模型名字，用来在存储时区分是哪个模型的')
    
    parser.add_argument('--max_samples', type=int, default=100,
                       help='最大样本数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小')
    parser.add_argument('--save_dir', type=str, default=f"figs/distance",
                       help='保存目录（默认根据数据类型自动设置）')

    args = parser.parse_args()

    args.save_dir = f"{args.save_dir}/{args.model_name}"

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model_base = construct_model_base(args.model_path, 'en')

    # 创建激活提取器
    extractor = ActivationExtractor(model_base)

    ############### 为每一种语言计算生成距离 ##################
    langs = ['en', 'zh', 'ja', 'ko', 'ar', 'bn', 'sw']
    for lang in langs:
        harmful_path = f"{args.harmful_dir}/harmful_test_translated_{lang}.json"
        harmless_path = f"{args.harmful_dir}/harmless_test_translated_{lang}.json"
        # 根据数据类型加载数据
        if args.data_type == 'dpo':
            harmless_texts, harmful_texts = load_dpo_data(args.data_path, args.max_samples)
            use_paired_distances = True  # DPO数据是配对的
        elif args.data_type == 'polyrefuse':
            harmless_texts, harmful_texts = load_polyrefuse_data(harmful_path, harmless_path, args.max_samples)
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

        print("Generating PCA plots and calculating distances...")
        num_layers= model_base.model.config.num_hidden_layers
        # for layer in tqdm(range(num_layers), desc="Processing layers"):
        for layer in tqdm(range(num_layers-1, num_layers), desc="Processing layers"):
            # 最后token的激活
            layer_activations_last = activations_last[:, layer, :]  # [n_samples, d_model]
            calculate_and_save_distances(layer_activations_last, all_labels, layer, "last", args.save_dir, lang,use_paired_distances)

            # 平均token的激活
            layer_activations_mean = activations_mean[:, layer, :]  # [n_samples, d_model]
            calculate_and_save_distances(layer_activations_mean, all_labels, layer, "mean", args.save_dir, lang, use_paired_distances)

        print(f"All plots saved to {args.save_dir}")

    ############### 为en和每一种目标语言绘制直方图和箱线图 ##################
    # langs = ['zh', 'ja', 'ko', 'ar', 'bn', 'sw']
    # for lang in langs:
    #     # 绘制距离分布的直方图和箱线图
    #     for position_type in ['mean']: # ['last', 'mean']
    #         plot_distance_hist_en_vs_other(
    #             args,
    #             layer=layer,
    #             position_type=position_type,
    #             other_lang=lang
    #         )

    #         plot_distance_box_en_vs_other(
    #             args,
    #             layer=layer,
    #             position_type=position_type,
    #             other_lang=lang
    #         )


    ############### 7 种语言同图（直方图 + 箱线图） ##################
    all_langs = ['en', 'zh', 'ja', 'ko', 'ar', 'bn', 'sw']
    for position_type in ['mean']:  # 或 ['last','mean']
        plot_distance_hist_all_langs(
            args,
            layer=layer,
            position_type=position_type,
            langs=all_langs,
            bins=60,
            figsize=(14, 4)
        )
        plot_distance_box_all_langs(
            args,
            layer=layer,
            position_type=position_type,
            langs=all_langs,
            figsize=(14, 4)
        )


if __name__ == "__main__":
    main()
