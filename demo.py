import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import random

from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def compute_motion_strength(flow):
    """Compute the average motion strength from optical flow.
    Args:
        flow: numpy array of shape (H, W, 2) containing flow vectors (u, v)
    Returns:
        float: average motion strength across all pixels
    """
    # Calculate magnitude for each pixel: sqrt(u^2 + v^2)
    magnitude = np.sqrt(np.sum(flow**2, axis=2))
    # Return average magnitude across all pixels
    return np.mean(magnitude)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # Compute motion strength
    motion_strength = compute_motion_strength(flo)
    
    # map flow to rgb image
    flo_viz = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo_viz], axis=0)

    # Create filename with current timestamp and random number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_num = random.randint(1000, 9999)  # 4-digit random number
    output_filename = f'flow_result_{timestamp}_{random_num}.png'
    
    # Save the image
    cv2.imwrite(output_filename, img_flo[:, :, [2,1,0]])
    print(f"Saved flow visualization to {output_filename}")
    print(f"Motion strength: {motion_strength:.4f}")

    
 


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    # python demo.py --model=data/raft/raft-things.pth --path=../RAFT/demo-frames
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
