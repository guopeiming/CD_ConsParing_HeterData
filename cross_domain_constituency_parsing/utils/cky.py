import numpy as np
from typing import List, Tuple


def CKY(logits: np.ndarray, len_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retures:
        pred_events: 4维ndarray，[batch_size, seq_len, seq_len, label_num]，对应span的对应标签是1，其余为0。
        tree_split_tables: 3维ndarray，[batch_size, seq_len, seq_len]，对应span的值是span的切分点。
    """
    # 1. 计算每个span对应位置的最大标签得分，
    # 存入 best_label_score_tables  (b, len, len) 和 best_label_index_tables (b, len, len)
    best_label_score_tables = np.max(logits, axis=-1)
    best_label_index_tables = np.argmax(logits, axis=-1)

    # 2. 计算每个span对应位置的最大总得分，
    # 存入 tree_score_tables (b, len, len) 并且保存最大得分对应的切分 tree_split_tables (b, len, len)
    tree_score_tables = np.zeros(best_label_score_tables.shape)
    tree_split_tables = np.full(best_label_score_tables.shape, -1)
    pred_events = np.zeros(logits.shape)

    for B, snt_len in enumerate(len_list):
        single_CKY(tree_score_tables[B], tree_split_tables[B], best_label_score_tables[B], snt_len)
        find_tree(tree_split_tables[B], best_label_index_tables[B], pred_events[B], 0, snt_len-1)

    return pred_events, tree_split_tables


def single_CKY(tree_score: np.ndarray, tree_split: np.ndarray, best_label_score: np.ndarray, snt_len: int):
    tree_score[np.arange(snt_len), np.arange(snt_len)] = np.diagonal(best_label_score, axis1=0)[:snt_len]

    for i in range(1, snt_len):
        for j in range(i, snt_len):
            span_start, span_end = j-i, j
            max_score, best_k = get_best_k_score(tree_score, span_start, span_end)
            tree_score[span_start, span_end] = max_score + best_label_score[span_start, span_end]
            tree_split[span_start, span_end] = best_k


def get_best_k_score(tree_score: np.ndarray, i: int, j: int) -> Tuple[float, int]:
    max_score, best_k = tree_score[i][i] + tree_score[i+1][j], i
    for k in range(i, j):
        l_score = tree_score[i][k] + tree_score[k+1][j]
        if l_score > max_score:
            max_score, best_k = l_score, k
    return max_score, best_k


def find_tree(
    tree_split: np.ndarray, best_label_index: np.ndarray, pred_event: np.ndarray, span_s: int, span_e: int
) -> None:
    pred_event[span_s, span_e, best_label_index[span_s, span_e]] = 1.

    k = tree_split[span_s, span_e]
    if k == -1:
        return
    else:
        find_tree(tree_split, best_label_index, pred_event, span_s, k)
        find_tree(tree_split, best_label_index, pred_event, k+1, span_e)
