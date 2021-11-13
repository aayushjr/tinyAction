import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import random
import numpy as np
from configuration import build_config
from tqdm import tqdm
import time


def resize(frames, size, interpolation='bilinear'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(frames.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(frames, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TinyVirat(Dataset):
    def __init__(self, cfg, data_split, data_percentage, num_frames, skip_frames, input_size, shuffle=False):
        self.data_split = data_split
        self.num_classes = cfg.num_classes
        self.class_labels = [k for k, v in sorted(json.load(open(cfg.class_map, 'r')).items(), key=lambda item: item[1])]
        assert data_split in ['train', 'val', 'test']
        if data_split == 'train':
            annotations = json.load(open(cfg.train_annotations, 'r'))
        elif data_split == 'val':
            annotations = json.load(open(cfg.val_annotations, 'r'))
        else:
            annotations = json.load(open(cfg.test_annotations, 'r'))
        self.data_folder = os.path.join(cfg.data_folder, data_split)
        self.annotations  = {}
        for annotation in annotations:
            if annotation['dim'][0] < num_frames:
                continue
            if annotation['id'] not in self.annotations:
                self.annotations[annotation['id']] = {}
            self.annotations[annotation['id']]['path'] = annotation['path']
            if data_split == 'test':
                self.annotations[annotation['id']]['label'] = []
            else:
                self.annotations[annotation['id']]['label'] = annotation['label']
            self.annotations[annotation['id']]['length'] = annotation['dim'][0]
            self.annotations[annotation['id']]['width'] = annotation['dim'][1]
            self.annotations[annotation['id']]['height'] = annotation['dim'][2]
        self.video_ids = list(self.annotations.keys())
        if shuffle:
            random.shuffle(self.video_ids)
        len_data = int(len(self.video_ids) * data_percentage)
        self.video_ids = self.video_ids[0:len_data]
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])

    def __len__(self):
        return len(self.video_ids)

    def load_frames_random(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip_frames = self.skip_frames
        while frame_count < self.num_frames * skip_frames:
            if skip_frames <= 1:
                skip_frames = 1
                break
            skip_frames = skip_frames // 2
        assert frame_count >= self.num_frames * skip_frames
        random_start = random.randint(0, frame_count - self.num_frames * skip_frames)
        frame_indicies = [indx for indx in range(random_start, random_start + self.num_frames * skip_frames, skip_frames)]
        ret = True
        counter = 0
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if counter > max(frame_indicies):
                ret = False
            if counter in frame_indicies:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                counter += 1
            else:
                counter += 1
                continue
        vidcap.release()
        assert len(frames) == self.num_frames
        frames = torch.from_numpy(np.stack(frames))
        return frames

    def load_all_frames(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = True
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        assert len(frames) == frame_count
        frames = torch.from_numpy(np.stack(frames))
        return frames

    def build_random_clip(self, video_path):
        frames = self.load_frames_random(video_path)
        frames = self.transform(frames)
        return frames

    def build_consecutive_clips(self, video_path):
        frames = self.load_all_frames(video_path)
        if len(frames) % self.num_frames != 0:
            frames = frames[:len(frames) - (len(frames) % self.num_frames)]
        clips = torch.stack([self.transform(x) for x in chunks(frames, self.num_frames)])
        return clips
    
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_path = os.path.join(self.data_folder, self.annotations[video_id]['path'])
        video_len = self.annotations[video_id]['length']
        
        if self.data_split == 'test':
            video_labels = []
        else:
            video_labels = self.annotations[video_id]['label']
        if self.data_split == 'train':
            clips = self.build_random_clip(video_path)
        else:
            clips = self.build_consecutive_clips(video_path)
            
            if self.data_split == 'test':
                return clips, [self.annotations[video_id]]
                
        label = np.zeros(self.num_classes)
        for _class in video_labels:
            label[self.class_labels.index(_class)] = 1
        return clips, label


if __name__ == '__main__':
    shuffle = True
    batch_size = 1

    dataset = 'TinyVirat'
    cfg = build_config(dataset)

    data_generator = TinyVirat(cfg, 'val', 1.0, num_frames=16, skip_frames=2, input_size=112)
    dataloader = DataLoader(data_generator, batch_size, shuffle=shuffle, num_workers=0)

    start = time.time()

    for epoch in range(0, 1):
        for i, (clips, labels) in enumerate(tqdm(dataloader)):
            clips = clips.data.numpy()
            labels = labels.data.numpy()
            print(clips.shape)
            print(labels.shape)
            break
    print("time taken : ", time.time() - start)
