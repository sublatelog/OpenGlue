import cv2
import kornia
import numpy as np
import torch
import torchmetrics

from .misc import normalize_with_intrinsics


class AccuracyUsingEpipolarDist(torchmetrics.Metric):
    def __init__(self, threshold=5e-4):
        super(AccuracyUsingEpipolarDist, self).__init__(dist_sync_on_step=True, compute_on_step=False)
        self.threshold = threshold
        self.add_state('precision', default=[], dist_reduce_fx='cat')
        self.add_state('matching_score', default=[], dist_reduce_fx='cat')

    def update(self, matched_kpts0, matched_kpts1, transformation, num_detected_kpts):
        K0 = transformation['K0']
        K1 = transformation['K1']
        R = transformation['R']
        T = transformation['T'].unsqueeze(-1)

        # essentialを取得
        E = kornia.geometry.epipolar.essential_from_Rt(
                                                       R1=torch.eye(3, device=R.device).unsqueeze(0),
                                                       t1=torch.zeros(1, 3, 1, device=R.device),
                                                       R2=R.unsqueeze(0), 
                                                       t2=T.unsqueeze(0)
                                                       )

        num_matched_kpts = matched_kpts0.shape[0]
        
        if num_matched_kpts > 0:
            matched_kpts0 = normalize_with_intrinsics(matched_kpts0, K0)
            matched_kpts1 = normalize_with_intrinsics(matched_kpts1, K1)

            # エピポーラー幾何を取得
            epipolar_dist = kornia.geometry.epipolar.symmetrical_epipolar_distance(
                                                                                    matched_kpts0.unsqueeze(0),
                                                                                    matched_kpts1.unsqueeze(0),
                                                                                    E
                                                                                    ).squeeze(0)
            # 閾値以下の値の合計
            num_correct_matches = (epipolar_dist < self.threshold).sum()
            
            # 全ペアに対する正解ペアの割合
            precision = num_correct_matches / num_matched_kpts
            
            # 全キーポイントに対する正解ペアの割合
            matching_score = num_correct_matches / num_detected_kpts
        else:
            precision, matching_score = matched_kpts0.new_tensor(0.), matched_kpts0.new_tensor(0.)
            
            
        self.precision.append(precision)
        self.matching_score.append(matching_score)

    # 全ペアに対する正解ペアの割合と全キーポイントに対する正解ペアの割合を返す
    def compute(self):
        
        return {
            'Precision': self.precision.mean(),
            'Matching Score': self.matching_score.mean()
        }


class CameraPoseAUC(torchmetrics.Metric):
    def __init__(self, auc_thresholds, ransac_inliers_threshold):
        super(CameraPoseAUC, self).__init__(
            dist_sync_on_step=True, compute_on_step=False,
        )
        self.auc_thresholds = auc_thresholds
        self.ransac_inliers_threshold = ransac_inliers_threshold

        self.add_state('pose_errors', default=[], dist_reduce_fx='cat')

    @staticmethod
    def __rotation_error(R_true, R_pred):
        """ 予測と正解のRの角度差 """
        
        """
        R_true
        tensor([[ 0.9961, -0.0290, -0.0830],
                [ 0.0283,  0.9996, -0.0088],
                [ 0.0833,  0.0064,  0.9965]], device='cuda:0')
                
        R_pred
        tensor([[-0.8351,  0.5485, -0.0429],
                [-0.5489, -0.8255,  0.1314],
                [ 0.0366,  0.1333,  0.9904]], device='cuda:0')
                
        R_true * R_pred
        tensor([[-8.3184e-01, -1.5889e-02,  3.5654e-03],
                [-1.5554e-02, -8.2512e-01, -1.1561e-03],
                [ 3.0487e-03,  8.5468e-04,  9.8694e-01]], device='cuda:0')
                
        (R_true * R_pred).sum()
        tensor(-0.6951, device='cuda:0')
        """

        
        # torch.arccos():転置cos()
        angle = torch.arccos(torch.clip(((R_true * R_pred).sum() - 1) / 2, -1, 1))
        
        # torch.rad2deg():ラジアン → 度
        return torch.abs(torch.rad2deg(angle))

    @staticmethod
    def __translation_error(T_true, T_pred):
        """ 予測と正解のTのコサイン類似度の角度差 """
        
        # torch.cosine_similarity():コサイン類似度は、ベクトルまたはテンソル、ユークリッド距離
        angle = torch.arccos(torch.cosine_similarity(T_true, T_pred, dim=0))[0]
        
        angle = torch.abs(torch.rad2deg(angle))
        return torch.minimum(angle, 180. - angle)

    def update(self, matched_kpts0, matched_kpts1, transformation):
        device = matched_kpts0.device
        K0 = transformation['K0']
        K1 = transformation['K1']
        R = transformation['R']
        T = transformation['T'].unsqueeze(-1)

        # estimate essential matrix from point matches in calibrated space
        num_matched_kpts = matched_kpts0.shape[0]
        
        # キーポイントが5個以上ある場合のみ実行
        if num_matched_kpts >= 5:
            
            # convert to calibrated space and move to cpu for OpenCV RANSAC
            matched_kpts0_calibrated = normalize_with_intrinsics(matched_kpts0, K0).cpu().numpy()
            matched_kpts1_calibrated = normalize_with_intrinsics(matched_kpts1, K1).cpu().numpy()

            threshold = 2 * self.ransac_inliers_threshold / (K0[[0, 1], [0, 1]] + K1[[0, 1], [0, 1]]).mean()
            
            # Essential Matrix (基本行列) を取得
            E, mask = cv2.findEssentialMat(
                                            matched_kpts0_calibrated,
                                            matched_kpts1_calibrated,
                                            np.eye(3),
                                            threshold=float(threshold),
                                            prob=0.99999,
                                            method=cv2.RANSAC
                                            )
            
            if E is None:
                error = torch.tensor(np.inf).to(device)
            else:
                E = torch.FloatTensor(E).to(device)
                mask = torch.BoolTensor(mask[:, 0]).to(device)

                best_solution_n_points = -1
                best_solution = None
                for E_chunk in E.split(3):
                    
                    # Essential Matrix (基本行列) からＲとＴを復元
                    R_pred, T_pred, points3d = kornia.geometry.epipolar.motion_from_essential_choose_solution(
                                                                                                                E_chunk, 
                                                                                                                K0, 
                                                                                                                K1,
                                                                                                                matched_kpts0, 
                                                                                                                matched_kpts1,
                                                                                                                mask=mask
                                                                                                              )
                    
                    # 3d point
                    n_points = points3d.size(0)
                    if n_points > best_solution_n_points:
                        best_solution_n_points = n_points
                        best_solution = (R_pred, T_pred)
                        
                R_pred, T_pred = best_solution

                # Rの角度誤差とTのコサイン類似度角度誤差を求める
                R_error, T_error = self.__rotation_error(R, R_pred), self.__translation_error(T, T_pred)
                
                # 各要素について両エラーで最大の値を返す
                error = torch.maximum(R_error, T_error)
                
        else:
            error = torch.tensor(np.inf).to(device)
            
        self.pose_errors.append(error)
        

    def compute(self):
        errors = self.pose_errors
        errors = torch.sort(errors).values
        
        """
        errors
        torch.Size([1])
        tensor([inf], device='cuda:0')
        """
      
        recall = (torch.arange(len(errors), device=errors.device) + 1) / len(errors)
        """
        recall
        tensor([1.], device='cuda:0')
        """
        
        zero = torch.zeros(1, device=errors.device)
        errors = torch.cat([zero, errors])
        recall = torch.cat([zero, recall])
        """
        errors
        tensor([0., inf], device='cuda:0')
        
        recall
        tensor([0., 1.], device='cuda:0')
        """

        aucs = {}
        # camera_auc_thresholds: [5.0, 10.0, 20.0]
        for threshold in self.auc_thresholds:
            threshold = torch.tensor(threshold).to(errors.device)
            
            """
            threshold
            tensor(5., device='cuda:0')
            """

            # thresholdに一番近い値のindexを取得
            last_index = torch.searchsorted(errors, threshold)
            
            """
            last_index
            tensor(1, device='cuda:0')
            """
            
            r = torch.cat([recall[:last_index], recall[last_index - 1].unsqueeze(0)])
            """
            r
            tensor([0., 0.], device='cuda:0')
            torch.Size([2])
            """
            
            e = torch.cat([errors[:last_index], threshold.unsqueeze(0)])
            
            """
            e
            tensor([0., 5.], device='cuda:0')
            torch.Size([2])
            """
            
            # torch.trapz():台形公式は定積分を近似計算するための方法、すなわち数値積分のひとつである。
            area = torch.trapz(r, x=e) / threshold
            
            """
            torch.trapz(r, x=e)
            tensor(0., device='cuda:0')
            torch.Size([])
            """
            
            aucs[f'AUC@{threshold}deg'] = area
            
        """
        aucs
        {'AUC@5.0deg': tensor(0., device='cuda:0'), 'AUC@10.0deg': tensor(0., device='cuda:0'), 'AUC@20.0deg': tensor(0., device='cuda:0')}
        """
        
        return aucs
