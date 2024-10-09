import argparse
import torch
import cv2
import numpy as np
# import files from PEGG-Net folder:
import PEGG_Net.models.peggnet as peggnet
# import custom files:
import utils.image_processing as ip


def parse_args():
    parser = argparse.ArgumentParser(description='Run a pretrained model on a RGBD or D picture')
    parser.add_argument('--model', type=str, required=False, help='Path to the model file')
    parser.add_argument('--state-dict', type=str, required=False, help='Path to the state dict file')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB image')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth image')
    parser.add_argument('--output', type=str, required=False, help='Path to the output file')
    parser.add_argument('--use-rgb', action='store_true',default=False ,help='Use RGB image as input')
    parser.add_argument('--use-depth', action='store_true',default=True , help='Use depth image as input')
    parser.add_argument('--vis', action='store_true',default=False , help='Visualize the output')

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    assert args.use_rgb or args.use_depth, 'At least one of RGB or depth must be used'
    
    # Determine the number of input channels based on the input type
    input_channels = 4
    if args.use_rgb and not args.use_depth:
        input_channels = 3
    elif args.use_depth and not args.use_rgb:
        input_channels = 1
    # Load the images
    rgb = cv2.imread(args.rgb, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)

    # Load the model
    assert args.model is not None or args.state_dict is not None, 'Either model or state-dict must be specified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    net = None
    if args.model is not None:
        print('Loading model from {}'.format(args.model))
        net = torch.load(args.model, map_location=device)
    else:
        print('Loading model from state dict {}'.format(args.state_dict))
        net = peggnet.PEGGNet(input_channels=input_channels)
        net.load_state_dict(torch.load(args.state_dict, map_location=device))
    assert net is not None, 'Model not loaded'
    print('Model loaded')
    
    

if __name__ == '__main__':
    main()

