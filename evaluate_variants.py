import os
import numpy as np
import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from dataloader import TinyVirat

from model import generate_model, load_pretrained_model
from opts import parse_opts

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json
import random
import argparse
from datetime import datetime

from configuration import build_config


def resume_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    print("Loaded model weights from ", resume_path)
    
    return model
    
    
def inference_epoch(cfg, data_loader, model, use_cuda, args, save_path):
    print('Inference')

    num_classes = cfg.num_classes

    model.eval()
    
    f1_threshold = 0.5
    
    # Writer
    with open(save_path,'w') as wid:
        
        vid_id = 0
        for i, (clips, video_id) in enumerate(data_loader):
            
            video_id = video_id[0]['path'][0]
            video_id = video_id.split('.')[0]
            
            while vid_id < int(video_id):
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
            
            if args.model == 'i3d':
                outputs = torch.max(outputs, dim=2)[0]
            outputs = torch.max(outputs, dim=0)[0]
            outputs = outputs.reshape(-1, num_classes).cpu().data.numpy()
            
            outputs[outputs<= f1_threshold] = 0
            outputs[outputs > f1_threshold] = 1
            
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
    print("Run ID : " + run_id)

    val_data_generator = TinyVirat(cfg, 'test', 1.0, num_frames=args.num_frames, skip_frames=args.skip_frames, input_size=args.sample_size)
    val_dataloader = DataLoader(val_data_generator, 1, shuffle=shuffle, num_workers=0)

    print("Number of eval samples : " + str(len(val_data_generator)))
    
    num_classes = cfg.num_classes

    model = generate_model(args)
    model = resume_model(args.pretrain_path, model)
    
    if use_cuda:
        model.cuda()
    
    save_path = os.path.join(save_dir, 'answer.txt')
    inference_epoch(cfg, val_dataloader, model, use_cuda, args, save_path)


def eval_classifier(run_id, use_cuda, args):
    cfg = build_config("TinyVirat")
    save_dir = os.path.join("./submissions", run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inference_model(cfg, run_id, save_dir, use_cuda, args)


def main(args):
    
    run_id = args.model + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
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


def get_mean_std(value_scale, dataset):
    assert dataset in ['tinyvirat', 'activitynet', 'kinetics', '0.5']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'tinyvirat':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]    

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std
    
    
def get_opt():
    opt = parse_opts()
    
    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
        
    opt.no_cuda = True if not torch.cuda.is_available() else False 
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    
    return opt
    

if __name__ == '__main__':

    args = get_opt()
    main(args)