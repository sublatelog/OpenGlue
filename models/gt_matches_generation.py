"""
Module for generating ground truth matches between two images given keypoints on both images
and ground truth transformation
"""

from typing import Dict, Any, Optional, Tuple

import torch

from utils.misc import get_inverse_transformation, reproject_keypoints

# define module constants
UNMATCHED_INDEX = -1  # index of keypoint that don't have a match
IGNORE_INDEX = -2  # index of keypoints to ignore during loss calculation


def generate_gt_matches(
                        data: Dict[str, Any],
                        features0: Dict[str, torch.Tensor],
                        features1: Dict[str, torch.Tensor],
                        positive_threshold: float,                    # gt_positive_threshold: 2  
                        negative_threshold: Optional[float] = None    # gt_negative_threshold: 7
                        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, torch.Tensor]]]:
  
    """Given image pair, keypoints detected in each image, return set of ground truth correspondences"""
    
    # ネガティブ閾値がない場合はポジティブ閾値を使う
    if negative_threshold is None:
        negative_threshold = positive_threshold

    # キーポイントを取り出す
    kpts0, kpts1 = features0['keypoints'], features1['keypoints']

    # transformationを取り出す
    transformation = data['transformation']
    
    # '3d_reprojection'タイプの場合はTに転置した'R'の行列積を与える。 default:'3d_reprojection'
    # transformationのTだけ変更
    transformation_inv = get_inverse_transformation(transformation)
    
    """
    kpts0
    torch.Size([1, 407, 2])
    tensor([[[309.5390, 219.1302],
             [344.5755, 216.3512],
             [326.8630, 217.9524],
             [207.5000, 511.5000],
             [328.0000, 169.1887],
             [432.2422, 515.2451],
             [349.5874, 222.8119],
             [367.3845, 515.3196],
             [340.7626, 227.1615],
    """
    
    """
    kpts1
    torch.Size([1, 639, 2])
    tensor([[[223.8122, 329.9168],
             [281.6742, 344.0130],
             [251.0376, 332.2923],
             ...,
             [119.0000,  70.0000],
             [450.2680, 592.3522],
             [185.7256, 755.5035]]], device='cuda:0')
    """
       
    
    # キーポイントの個数を取得
    num0, num1 = kpts0.size(1), kpts1.size(1)
    """
    num0
    674
    num1
    639
    """

    # skip step if no keypoint are detected
    if num0 == 0 or num1 == 0:
        return None, None

    
    # GTの変換でキーポイントを相対的な正解位置へ動かす
    # establish ground truth correspondences given transformation
    kpts0_transformed, mask0 = reproject_keypoints(kpts0, transformation)
    kpts1_transformed, mask1 = reproject_keypoints(kpts1, transformation_inv)
    
    
    
    # torch.cdist:ノルム距離を計算　kpts0_transformed == kpts1　正解位置と対称予測位置
    reprojection_error_0_to_1 = torch.cdist(kpts0_transformed, kpts1, p=2)  # batch_size x num0 x num1
    
    # torch.cdist:ノルム距離を計算　kpts1_transformed == kpts0　正解位置と対称予測位置
    reprojection_error_1_to_0 = torch.cdist(kpts1_transformed, kpts0, p=2)  # batch_size x num1 x num0

    # ノルム距離誤差の行単位での最小値 ([1, 407, 342]) > ([1, 407])
    min_dist0, nn_matches0 = reprojection_error_0_to_1.min(2)  # batch_size x num0 (min, min_indices)
    min_dist1, nn_matches1 = reprojection_error_1_to_0.min(2)  # batch_size x num1
    
    """
    reprojection_error_0_to_1
    torch.Size([1, 407, 342])
    tensor([[[888.9370, 729.9428, 737.5640,  ..., 774.6237, 776.7797, 689.4753],
             [666.5838, 526.7121, 515.1260,  ..., 562.4525, 563.9073, 466.7016],
             [807.1566, 644.3197, 656.0607,  ..., 705.9921, 707.6750, 610.2487],
             ...,
             [641.8162, 585.6963, 497.5612,  ..., 460.9817, 464.5255, 433.3936],
             [723.5299, 657.9354, 578.7091,  ..., 538.3829, 542.2457, 515.0137],
             [553.9631, 517.8069, 411.9433,  ..., 373.6161, 376.8466, 346.2145]]],
           device='cuda:0')

    min_dist0:min
    torch.Size([1, 407])
    tensor([[495.2804, 280.5597, 410.6893, 264.6145, 313.4991, 508.4745, 509.9016,
             364.5308, 512.9706, 205.5891, 249.4042, 437.8380, 202.8465, 389.0567,
             276.8835, 546.8709, 343.2892, 258.0063, 375.0814, 388.5355, 509.7009,
             895.2272, 297.5146, 285.4641, 331.8199, 290.5402, 351.5373, 284.0844,
             388.3299, 218.4801, 270.1385, 281.4160, 379.6379, 167.4273, 161.6492,
             175.9752, 333.6596, 537.4679, 269.9494, 279.3772, 295.9263, 281.1237,
             373.3811, 302.3147, 367.6788, 305.8423, 250.9976, 268.4193, 204.4420,
    
    
    nn_matches0:min_indices
    torch.Size([1, 407])
    tensor([[ 76,  76,  76, 243, 243,  76,  76, 243,  76,  76, 243,  76, 243, 243,
              76,  76,  76, 243, 243, 243,  76,  76, 243, 243, 243, 243,  76, 243,
             243, 243,  76, 243, 243, 243,  76, 243, 243,  76, 243,  76, 243, 243,
             243, 243, 243, 243, 243, 243, 243,  76, 243, 243, 243, 243, 243, 243,
             243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243,
             243, 243, 243, 243,  76, 243, 243, 243, 243, 243, 243,  76, 243, 243,
  
    nn_matches1
    torch.Size([1, 639])
    tensor([[371, 154, 154, 154, 356, 154, 371, 627, 627, 356, 154, 154, 154, 305,
             667, 627, 154, 305, 305, 305, 127, 305, 371, 371, 627, 131, 627, 154,
             667, 154, 154, 627, 131, 627, 627, 356, 667, 154, 667, 305, 667, 627,
             627, 371, 305, 667, 131, 667, 627, 627, 154, 667, 127, 305, 667, 154,
              30,  30, 127, 627, 667, 154, 627, 627, 371, 356, 371, 667, 667, 627,
             627, 305, 627, 627, 627, 127, 667, 356, 667, 154, 667, 305, 627, 667,
             627, 627, 627, 667, 667, 627, 363, 627, 153]], device='cuda:0')

    """

    
    # 複製   
    gt_matches0, gt_matches1 = nn_matches0.clone(), nn_matches1.clone()
    device = gt_matches0.device
    
    # gt_matches0の行最小値列indexでgt_matches1の
    cross_check_consistent0 = torch.arange(num0, device=device).unsqueeze(0) == gt_matches1.gather(1, gt_matches0)
    gt_matches0[~cross_check_consistent0] = UNMATCHED_INDEX # -1
    """
    torch.arange(num0, device=device).unsqueeze(0)
    torch.Size([1, 787])
    tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
              14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
              ~
              770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783,
              784, 785, 786]], device='cuda:0')
              
    gt_matches1.gather(1, gt_matches0)
    torch.Size([1, 787])
    tensor([[502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 677,
             502, 502, 502, 502, 502, 502, 502, 677, 502, 677, 502, 502, 502, 677,
             502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 677, 677, 502,
             502, 677, 502, 502, 677, 677, 502, 502, 502, 502, 502, 502, 502, 502,          
             502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502,
             502, 502, 502]], device='cuda:0')
             
    cross_check_consistent0
    torch.Size([1, 674])
    tensor([[False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False,  True, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,          
              
    """
    


    cross_check_consistent1 = torch.arange(num1, device=device).unsqueeze(0) == gt_matches0.gather(1, gt_matches1)
    gt_matches1[~cross_check_consistent1] = UNMATCHED_INDEX

    # so far mutual NN are marked MATCHED and non-mutual UNMATCHED

    # 両方の始点からの距離の平均
    symmetric_dist = 0.5 * (min_dist0[cross_check_consistent0] + min_dist1[cross_check_consistent1])

    # 平均距離がポジティブ閾値以上の場合は
    gt_matches0[cross_check_consistent0][symmetric_dist > positive_threshold] = IGNORE_INDEX # -2 # positive_threshold:2
    
    # 平均距離がネガティブ閾値以上
    gt_matches0[cross_check_consistent0][symmetric_dist > negative_threshold] = UNMATCHED_INDEX # negative_threshold:7

    gt_matches1[cross_check_consistent1][symmetric_dist > positive_threshold] = IGNORE_INDEX
    gt_matches1[cross_check_consistent1][symmetric_dist > negative_threshold] = UNMATCHED_INDEX

    gt_matches0[~cross_check_consistent0][min_dist0[~cross_check_consistent0] <= negative_threshold] = IGNORE_INDEX
    gt_matches1[~cross_check_consistent1][min_dist1[~cross_check_consistent1] <= negative_threshold] = IGNORE_INDEX

    # mutual NN with sym_dist <= pos.th ==> MATCHED
    # mutual NN with  pos.th < sym_dist <= neg.th ==> IGNORED
    # mutual NN with neg.th < sym_dist => UNMATCHED
    # non-mutual with dist <= neg.th ==> IGNORED
    # non-mutual with dist > neg.th ==> UNMATCHED

    # ignore kpts with unknown depth data
    gt_matches0[~mask0] = IGNORE_INDEX
    gt_matches1[~mask1] = IGNORE_INDEX

    # also ignore MATCHED point if its nearest neighbor is invalid
    gt_matches0[cross_check_consistent0][~mask1.gather(1, nn_matches0)[cross_check_consistent0]] = IGNORE_INDEX
    gt_matches1[cross_check_consistent1][~mask0.gather(1, nn_matches1)[cross_check_consistent1]] = IGNORE_INDEX

    data = {
        **data,
        'keypoints0': kpts0, 
        'keypoints1': kpts1,
        'local_descriptors0': features0['local_descriptors'], 
        'local_descriptors1': features1['local_descriptors'],
        'side_info0': features0['side_info'], 
        'side_info1': features1['side_info'],
    }
    
    
#     print("gt_matches0")
#     print(gt_matches0)
#     print(gt_matches0.shape)
    
    
    print("gt_matches1")
    print(gt_matches1)
    print(gt_matches1.shape)
    

    y_true = {
        'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1
    }

    return data, y_true
