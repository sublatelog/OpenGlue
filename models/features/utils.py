import inspect

import kornia.feature as KF
import numpy as np
import torch


def filter_dict(dict_to_filter, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict


def get_descriptors(img, descriptor, lafs=None, patch_size=32):
    r"""Acquire descriptors for each keypoint given an original image, its keypoints, and a descriptor module"""

    patches = KF.extract_patches_from_pyramid(img, lafs, patch_size)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

    return descs


def min_stack(keypoints, side_info, descriptors):
    """
    Stack batch of keypoints prediction into single tensor.
    For each instance keep number of keypoints minimal in the batch.
    """
    kpts_num = np.array([x.shape[0] for x in keypoints])
    min_kpts_to_keep = kpts_num.min()

    if np.all(kpts_num == min_kpts_to_keep):
        return torch.stack(keypoints), torch.stack(side_info), torch.stack(descriptors)
    else:
        # get scores and indices of keypoints to keep in each batch element
        indices_to_keep = [torch.topk(side_info_, min_kpts_to_keep, dim=0).indices
                           for side_info_ in side_info]

        data_stacked = {'keypoints': [], 'side_info': [], 'descriptors': []}
        for kpts, descs, info, idxs in zip(keypoints, descriptors, side_info, indices_to_keep):
            data_stacked['side_info'].append(info[idxs])
            data_stacked['keypoints'].append(kpts[idxs])
            data_stacked['descriptors'].append(descs[idxs])

        keypoints = torch.stack(data_stacked['keypoints'])
        side_info = torch.stack(data_stacked['side_info'])
        descriptors = torch.stack(data_stacked['descriptors'])

        return keypoints, side_info, descriptors

# extractor(SIFT)形式の出力をSuperGlue形式に変換
def prepare_features_output(lafs, responses, desc, laf_converter, permute_desc=False, log_response=False):
    
    # SuperGlueで受け取れる形に修正する
    """Convert features output into format acceptable by SuperGlue"""
    
    kpts = lafs[:, :, :, -1]
    responses = responses.unsqueeze(-1)
    
    # log変換
    if log_response:
        responses = (responses + 0.1).log()
    
    """
    lafs
    tensor([[[[-5.5220e+00,  3.1881e+00,  4.6020e+01],
              [-3.1881e+00, -5.5220e+00,  2.0565e+02]],

             [[-1.3255e+01, -4.8246e+00,  2.1576e+01],
              [ 4.8246e+00, -1.3255e+01,  3.6795e+02]],

             [[-1.3744e+01,  2.4234e+00,  5.1299e+01],
              [-2.4234e+00, -1.3744e+01,  2.0136e+02]],

             ...,

             [[-3.2834e-06, -8.1811e+00,  1.9791e+02],
              [ 8.1811e+00, -3.2834e-06,  5.9049e+02]],

             [[ 1.1218e+00,  6.3621e+00,  1.9310e+02],
              [-6.3621e+00,  1.1218e+00,  1.7341e+02]],

             [[-1.5008e+01, -2.6462e+00,  1.1900e+02],
              [ 2.6462e+00, -1.5008e+01,  2.5400e+02]]]], device='cuda:0')
    """
    
    print("lafs")
    print(lafs.shape)
    
    """
    laf_converter(lafs)
    tensor([], device='cuda:0', size=(1, 352, 0))
    """
    
    # laf_converter(lafs):lafs.new_empty(B, N, 0)    
    side_info = torch.cat([responses, laf_converter(lafs)], dim=-1)
    
    """
    kpts
    tensor([[[ 46.0198, 205.6502],
             [ 21.5761, 367.9525],
             [ 51.2985, 201.3564],
             [ 44.4080, 419.0637],
             [301.2976, 324.4242],
             [ 38.7864, 418.0502],
             [ 50.7517, 279.0307],
             [ 41.6554, 357.0193],
             [ 41.1325, 294.4109],
             [ 51.0867, 224.1311],
    """
    print("kpts")
    print(kpts.shape)
    
    print("desc")
    print(desc.shape)
    """
    desc
    tensor([[[0.2047, 0.0247, 0.2047,  ..., 0.0504, 0.0615, 0.0076],
             [0.1352, 0.0392, 0.0912,  ..., 0.0385, 0.0528, 0.0440],
             [0.0300, 0.1804, 0.1788,  ..., 0.0489, 0.0453, 0.0667],
             ...,
             [0.0113, 0.0461, 0.2047,  ..., 0.1655, 0.2047, 0.0554],
             [0.0278, 0.0283, 0.1864,  ..., 0.0445, 0.1081, 0.0657],
             [0.0564, 0.1971, 0.1322,  ..., 0.0780, 0.0623, 0.0727]]],
           device='cuda:0')
    """
    
    """
    desc.permute(0, 2, 1)
    tensor([[[0.2047, 0.1352, 0.0300,  ..., 0.0113, 0.0278, 0.0564],
             [0.0247, 0.0392, 0.1804,  ..., 0.0461, 0.0283, 0.1971],
             [0.2047, 0.0912, 0.1788,  ..., 0.2047, 0.1864, 0.1322],
             ...,
             [0.0504, 0.0385, 0.0489,  ..., 0.1655, 0.0445, 0.0780],
             [0.0615, 0.0528, 0.0453,  ..., 0.2047, 0.1081, 0.0623],
             [0.0076, 0.0440, 0.0667,  ..., 0.0554, 0.0657, 0.0727]]],
           device='cuda:0')
    """
    
    return {
        'keypoints': kpts,
        'side_info': side_info,
        'local_descriptors': desc.permute(0, 2, 1) if permute_desc else desc
    }
