import os
import numpy as np
import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from dataloader import TinyVirat
from models.pytorch_i3d_112 import InceptionI3d
from torchvision.models.video import r3d_18
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json
import random
import argparse
from datetime import datetime

from configuration import build_config


def build_model(backbone, num_classes, saved_model):
    if backbone == 'I3D':
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(num_classes)
    elif backbone == 'ResNet18':
        model = r3d_18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
                
    model.load_state_dict(torch.load(saved_model)['state_dict'])
    print("Loaded model weights from ", args.pretrained)
    
    return model
    
    
def inference_epoch(cfg, data_loader, model, use_cuda, args, save_path):
    print('Inference')

    num_gpus = len(args.gpu.split(','))

    num_classes = cfg.num_classes

    model.eval()
    
    # Writer
    with open(save_path,'w') as wid:
        
        vid_id = 0
        for i, (clips, video_id) in enumerate(data_loader):
            
            video_id = video_id[0]['path'][0]
            video_id = video_id.split('.')[0]
            
            if vid_id < int(video_id):
                empty_string = "{:05d} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".format(vid_id)
                #print(empty_string)
                wid.write(empty_string+'\n')
                vid_id += 1
            
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))

            with torch.no_grad():
                clips = clips.squeeze(0)
                outputs = model(clips)
                outputs = torch.sigmoid(outputs)
            
            outputs = torch.max(outputs, dim=0)[0]
            outputs = outputs.reshape(-1, num_classes).cpu().data.numpy()
            
            outputs[outputs<= args.f1_threshold] = 0
            outputs[outputs > args.f1_threshold] = 1
            
            result_string = "{}".format(video_id)
            for j in range(num_classes):
                result_string = "{} {}".format(result_string, int(outputs[0,j]))
            
            vid_id += 1
            #print(result_string)
            wid.write(result_string+'\n')
            
            if i%1000 == 0:
                print("{}/{}".format(i, len(data_loader)))
        
        while vid_id < 6097:
            empty_string = "{:05d} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".format(vid_id)
            wid.write(empty_string+'\n')
            vid_id += 1

def inference_model(cfg, run_id, save_dir, use_cuda, args):
    shuffle = False
    print("Run ID : " + args.run_id)

    val_data_generator = TinyVirat(cfg, 'test', 1.0, num_frames=args.num_frames, skip_frames=args.skip_frames, input_size=args.input_size)
    val_dataloader = DataLoader(val_data_generator, 1, shuffle=shuffle, num_workers=0)

    print("Number of eval samples : " + str(len(val_data_generator)))
    
    num_classes = cfg.num_classes

    model = build_model(args.backbone, num_classes, args.pretrained)
    
    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if use_cuda:
        model.cuda()
    
    save_path = os.path.join(save_dir, 'answer.txt')
    inference_epoch(cfg, val_dataloader, model, use_cuda, args, save_path)


def eval_classifier(run_id, use_cuda, args):
    cfg = build_config(args.dataset)
    save_dir = os.path.join("./submissions", run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inference_model(cfg, run_id, save_dir, use_cuda, args)


def main(args):
    print("Run description : ", args.run_description)

    # call a function depending on the 'mode' parameter
    if args.eval_classifier:
        run_id = args.run_id + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
        use_cuda = torch.cuda.is_available()
        eval_classifier(run_id, use_cuda, args)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to Evaluate TinyVirat Multi-label Classification model')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--eval_classifier', dest='eval_classifier', action='store_true', help='Training the Classifier')

    parser.add_argument("--gpu", dest='gpu', type=str, required=False, help='Set CUDA_VISIBLE_DEVICES environment variable, optional')

    parser.add_argument('--run_id', dest='run_id', type=str, required=False, help='Please provide an ID for the current run')

    parser.add_argument('--run_description', dest='run_description', type=str, required=False, help='Please description of the run to write to log')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use.', choices=["TinyVirat"])

    parser.add_argument('--backbone', type=str, required=True, help='Model type.', choices=["I3D", "ResNet18"])
    
    parser.add_argument('--pretrained', type=str, required=True, help='Trained model weights (pth) file.')

    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames in the input sequence.')

    parser.add_argument('--input_size', type=int, default=112, help='Spatial resolution of each frame in the input sequence.')

    parser.add_argument('--skip_frames', type=int, default=1, help='Number of frames to skip inbetween while building clips.')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')

    parser.add_argument('--f1_threshold', type=float, default=0.5, help='Probability threshold for computing F1 Score')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in the dataloader.')

    # parse arguments
    args = parser.parse_args()

    # set environment variables to use GPU-0 by default
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # exit when the mode is 'eval_classifier' and the parameter 'run_id' is missing
    if args.eval_classifier:
        if args.run_id is None:
            parser.print_help()
            exit(1)

    main(args)