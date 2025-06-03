import torch
from torch_geometric.data import Data
import json
import numpy as np

def find_approximate_key(feature, feature_dict, tolerance=1e-6):
    """
    在字典中查找与 feature 近似相等的键（基于欧氏距离）
    :param feature: 目标元组（浮点数）
    :param feature_dict: 待搜索的字典
    :param tolerance: 允许的最大欧氏距离
    :return: 匹配的键或 None
    """
    for key in feature_dict.keys():
        if np.allclose(feature, key, atol=tolerance):
            return key
    return None  # 未找到匹配项

def align_data_dpr_to_data(data, data_dpr, targeted, data_name)->Data.edge_index:
    data_name = data_name.lower()
    # Step 1: Extract features and labels
    x = data.x
    y = data.y
    x_dpr = data_dpr.x
    y_dpr = data_dpr.y

    # Step 2: Match nodes by features (exact match)
    # Create a dictionary: {feature_tuple: original_index_in_data_dpr}
    feature_and_label_to_index_origin = {}
    for i in range(x_dpr.shape[0]):
        feature_and_label = x_dpr[i].tolist()
        # feature_and_label.append(y_dpr[i].item())
        feature_and_label = tuple(np.round(feature_and_label, decimals=5))
        if feature_and_label in feature_and_label_to_index_origin:
            continue
        feature_and_label_to_index_origin[feature_and_label] = i

    # Step 3: Build mapping from data's nodes to data_dpr's nodes
    mapping = {}  # maps data's node index -> data_dpr's node index
    missing_nodes = []
    for i in range(x.shape[0]):
        feature = x[i].tolist()
        # feature.append(y[i].item())
        feature = tuple(np.round(feature, decimals=5))
        if feature in feature_and_label_to_index_origin:
            mapping[i] = (feature_and_label_to_index_origin[feature])
        else:
            missing_nodes.append(i)

    # if missing_nodes:
    #     print(f"Warning: {len(missing_nodes)} nodes in 'data' are missing in 'data_dpr'.")
    #     pass

    feature_and_label_to_index_origin = {}
    missing_nodes1 = []
    mapping1 = {}
    for i in range(x.shape[0]):
        feature_and_label = x[i].tolist()
        # feature_and_label.append(y[i].item())
        feature_and_label = tuple(np.round(feature_and_label, decimals=5))
        if feature_and_label in feature_and_label_to_index_origin:
            pass
        feature_and_label_to_index_origin[feature_and_label] = i

    for i in range(x_dpr.shape[0]):
        feature = x_dpr[i].tolist()
        # feature.append(y_dpr[i].item())
        feature = tuple(np.round(feature, decimals=5))
        if feature in feature_and_label_to_index_origin:
            mapping1[i] = (feature_and_label_to_index_origin[feature])
        else:
            missing_nodes1.append(i)

    # Step 4: Reindex edge_index
    edge_index_dpr = data_dpr.edge_index
    edge_index = data.edge_index

    # Create a reverse mapping: {old_idx_in_data_dpr -> new_idx_in_aligned_data}
    # reverse_mapping = {value: key for key, value in mapping.items()}
    reverse_mapping = mapping1
    missing_nodes1 = [int(i) for i in missing_nodes1]
    missing_nodes1 = set(missing_nodes1)

    edge_index_dpr0 = edge_index_dpr[0].tolist()
    edge_index_dpr1 = edge_index_dpr[1].tolist()

    # Reindex edges: replace old node indices with new ones
    existing_nodes0 = [i for i in edge_index_dpr0 if i not in missing_nodes1]
    existing_nodes1 = [i for i in edge_index_dpr1 if i not in missing_nodes1]
    new_row = torch.tensor([reverse_mapping[int(i)] for i in existing_nodes0], dtype=torch.long)
    new_col = torch.tensor([reverse_mapping[int(j)] for j in existing_nodes1], dtype=torch.long)
    new_edge_index = torch.stack([new_row, new_col])

    # Step 5: Filter edges involving missing nodes (if any)
    # missing_edges_mask = torch.tensor([
    # (src.item() in missing_nodes) or (dst.item() in missing_nodes)
    #     for src, dst in zip(edge_index[0], edge_index[1])
    # ], dtype=torch.bool)

    # missing_edges = edge_index[:, missing_edges_mask]

    # # Step 3: 合并两部分边（missing_edges + reindexed_dpr_edges）
    # final_edge_index = torch.cat([missing_edges, new_edge_index], dim=1)

    # Step 4: （可选）去重（如果合并后可能有重复边）
    final_edge_index = torch.unique(new_edge_index, dim=1)
    # final_edge_index = torch.unique(final_edge_index, dim=1)

    if targeted:
        path = f"/data/fusike/attacked graphs/nettack/{data_name}_nettacked_nodes.json"
        with open(path, "r") as fp:
            targeted_nodes = json.load(fp)
        targeted_nodes = targeted_nodes['attacked_test_nodes']
        targeted_nodes = [mapping1[node] for node in targeted_nodes]
    else:
        targeted_nodes = None
    return final_edge_index, targeted_nodes

# Usage:
# aligned_data_dpr = align_data_dpr_to_data(data, data_dpr)