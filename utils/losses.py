import numpy as np
import torch

from .misc import pairwise_cosine_dist


def criterion(y_true, y_pred, margin=None):
    # 正解
    gt_matches0, gt_matches1 = y_true['gt_matches0'], y_true['gt_matches1']
    
    # 予測
    gdesc0, gdesc1, scores = y_pred['context_descriptors0'], y_pred['context_descriptors1'], y_pred['scores']
    
    
    # ペア有triplet_loss ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    
    # marginを認める場合は2つの画像のgdescのcosine距離を計算
    if margin is not None:
        dist = pairwise_cosine_dist(gdesc0.transpose(2, 1).contiguous(), gdesc1.transpose(2, 1).contiguous())
    else:
        dist = None

    # loss for keypoints with gt match
    # 信頼度が0以上のキーポイントを取得
    batch_idx, idx_kpts0 = torch.where(gt_matches0 >= 0)    
    idx_kpts1 = gt_matches0[batch_idx, idx_kpts0]
    
    # torch.unique_consecutive():連続する重複値のみを削除する
    # 一意のリスト, index, 重複個数
    _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
    
    print("(1 / counts)")
    print((1 / counts))
    
    # 各キーポイントの重み
    mean_weights = (1 / counts)[inv_idx]

    # スコアを重み付けして合算
    matched_loss = (-scores[batch_idx, idx_kpts0, idx_kpts1] * mean_weights).sum()
    
    # triplet_loss
    matched_triplet_loss = matched_triplet_criterion(
                                                    dist=dist, 
                                                    margin=margin,
                                                    indexes=(batch_idx, idx_kpts0, idx_kpts1),
                                                    mean_weights=mean_weights
                                                    )
    

    # 画像0のペア無キーポイントのloss ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # loss for unmatched keypoints in the image 0
    batch_idx, idx_kpts0 = torch.where(gt_matches0 == -1)
    
    # torch.unique_consecutive():連続する重複値のみを削除する    
    # 一意のリスト, index, 重複個数
    _, inv_idx, counts = torch.unique_consecutive(
                                                  batch_idx, 
                                                  return_inverse=True, 
                                                  return_counts=True
                                                  )
    # 重み
    mean_weights = (1 / counts)[inv_idx]

    unmatched0_loss = (-scores[batch_idx, idx_kpts0, -1] * mean_weights).sum()
    
    unmatched0_margin_loss = unmatched_margin_criterion(
                                                        dist=dist, 
                                                        margin=margin, 
                                                        indexes=(batch_idx, idx_kpts0),
                                                        mean_weights=mean_weights, 
                                                        zero_to_one=True
                                                        )
    
    
    # 画像1のペア無キーポイントのloss ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # loss for unmatched keypoints in the image 1
    batch_idx, idx_kpts1 = torch.where(gt_matches1 == -1)
    _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
    mean_weights = (1 / counts)[inv_idx]

    unmatched1_loss = (-scores[batch_idx, -1, idx_kpts1] * mean_weights).sum()
    
    unmatched1_margin_loss = unmatched_margin_criterion(
                                                        dist=dist, 
                                                        margin=margin, 
                                                        indexes=(batch_idx, idx_kpts1),
                                                        mean_weights=mean_weights, 
                                                        zero_to_one=False
                                                        )

    return {
        'loss': (matched_loss + 0.5 * (unmatched0_loss + unmatched1_loss)) / scores.size(0),
        'metric_loss': (matched_triplet_loss + unmatched0_margin_loss + unmatched1_margin_loss) / scores.size(0)
    }


# ポジティブキーポイントとネガティブキーポイントとの差に重みを掛けてlossに設定
def matched_triplet_criterion(dist, margin, indexes, mean_weights):
    
    # marginを認めない場合は0を返す
    if margin is None:
        return torch.tensor(0, device=mean_weights.device)
    
    # marginを認める場合 ↓↓↓
    
    batch_idx, idx_kpts0, idx_kpts1 = indexes
    
    # distance between anchor and positive
    dist_ap = dist[batch_idx, idx_kpts0, idx_kpts1]

    dist_detached = dist.detach().clone()
    
    # np.infを設定
    dist_detached[batch_idx, idx_kpts0, idx_kpts1] = np.inf
    
    # 最小値のindexを取得
    idx_kpts0_closest_to_1 = torch.argmin(dist_detached, dim=1)
    idx_kpts1_closest_to_0 = torch.argmin(dist_detached, dim=2)
    
    # 最小値のキーポイントを取得
    idx_kpts1_neg = idx_kpts1_closest_to_0[batch_idx, idx_kpts0]
    idx_kpts0_neg = idx_kpts0_closest_to_1[batch_idx, idx_kpts1]

    # 最小値のキーポイントとの距離を取得
    dist_an0 = dist[batch_idx, idx_kpts0, idx_kpts1_neg]
    dist_an1 = dist[batch_idx, idx_kpts0_neg, idx_kpts1]

    # ポジティブキーポイントとネガティブキーポイントとの差をlossに設定
    loss0 = torch.maximum(dist_ap - dist_an0 + margin, torch.tensor(0, device=dist.device))
    loss1 = torch.maximum(dist_ap - dist_an1 + margin, torch.tensor(0, device=dist.device))
    
    # 各ロスに重みを掛けて合算
    return (loss0 * mean_weights).sum() + (loss1 * mean_weights).sum()


# 各キーポイントからペア無キーポイントまでの距離の最大値をlossに設定
def unmatched_margin_criterion(dist, margin, indexes, mean_weights, zero_to_one=True):
    
    # marginを認めない場合は0を返す    
    if margin is None:
        return torch.tensor(0, device=mean_weights.device)
    
    
    # marginを認める場合 ↓↓↓
    
    batch_idx, idx_kpts = indexes

    # 最小値のindexを取得
    idx_kpts_closest = torch.argmin(dist, dim=2 if zero_to_one else 1)
    
    # 最小値のキーポイントを取得
    idx_kpts_neg = idx_kpts_closest[batch_idx, idx_kpts]

    # distance anchor-negative
    if zero_to_one:
        # 各キーポイントからのペア無キーポイントまでの距離を取得
        dist_an = dist[batch_idx, idx_kpts, idx_kpts_neg]
    else:
        dist_an = dist[batch_idx, idx_kpts_neg, idx_kpts]

    # ペア無キーポイントまでの距離の最大値をlossに設定
    loss = torch.maximum(-dist_an + margin, torch.tensor(0, device=dist.device))
    return (loss * mean_weights).sum()
