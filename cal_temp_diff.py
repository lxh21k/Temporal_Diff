import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import mmcv

from flow_models.spynet import SpyNet
from flow_models.raft_core.raft import RAFT
from flow_models.raft_core.utils.utils import InputPadder

from utils.flow_warp import flow_warp
from utils.file_client import load_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_model', help='select optical flow model. support: spynet, raft')
    args = parser.parse_args()

    # Load image pair
    data_gt_root = './data/ldv_v2/test_gt/'
    data_lq_root = './data/ldv_v2/test_lq/'
    ref_path = data_gt_root + '015/f002.png'
    supp_path = data_gt_root + '015/f003.png'

    input_ref = load_img(ref_path)
    input_ref = input_ref.unsqueeze(0)
    input_supp = load_img(supp_path)
    input_supp = input_supp.unsqueeze(0)

    padder = InputPadder(input_ref.shape)
    input_ref, input_supp = padder.pad(input_ref, input_supp)

    # Initialize optical flow model
    if args.flow_model == 'spynet':
        pretrained = './checkpoints/spynet_20210409-c6c1bd09.pth'
        flow = SpyNet(pretrained)
    elif args.flow_model == 'raft':
        pretrained = './checkpoints/raft_models/raft-things.pth'
        flow = torch.nn.DataParallel(RAFT(small=False, mixed_percision=True, alternate_corr=False))
        print("Loading raft pretrain model...")
        flow.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu')))
    else:
        raise TypeError('Unsupported flow model. Please use spynet or raft.')
    
    flow.eval()
    
    with torch.no_grad():
        flow = flow(input_ref, input_supp)

        # Visualize optical flow
        flow_np = flow.permute(0, 2, 3, 1).squeeze()
        flow_np = flow_np.numpy()
        flow_map = np.uint8(mmcv.flow2rgb(flow_np) * 255.)
        plt.imsave('./gt_flow.png', flow_map)

        # Warp ref frame
        aligned_ref = flow_warp(input_ref, flow.permute(0, 2, 3, 1))
        aligned_ref = aligned_ref.squeeze()
        aligned_ref_np = aligned_ref.numpy()
        aligned_ref_np = np.transpose(aligned_ref_np[[2,1,0], :, :], (1, 2, 0))    # rgb to bgr & CHW to HWC
        aligned_ref_np = np.clip(aligned_ref_np, 0, 1.)

        plt.imsave('./aligned_ref.png', aligned_ref_np)
        print(aligned_ref_np.shape)

        # Compute temporal residual
        temp_diff = input_supp.squeeze() - aligned_ref
        # temp_diff = input_supp.squeeze() - input_ref.squeeze()
        temp_diff_np = temp_diff.numpy()
        temp_diff_np = np.transpose(temp_diff_np[[2,1,0], :, :], (1, 2, 0))
        # temp_diff_np = 0.5 * (temp_diff_np + 1.)
        temp_diff_np = abs(temp_diff_np)
        plt.imsave('./temp_diff.png', temp_diff_np)

