import argparse
import torch
# import file from PEGG-Net folder:


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
    assert args.model is not None or args.state_dict is not None, 'Either model or state-dict must be specified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = None
    if args.model is not None:
        print('Loading model from {}'.format(args.model))
        net = torch.load(args.model, map_location=device)
    else:
        print('Loading model from state dict {}'.format(args.state_dict))
        net 
    

if __name__ == '__main__':
    main()

