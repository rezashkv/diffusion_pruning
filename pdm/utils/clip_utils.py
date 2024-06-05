# Credits: https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import clip
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

TEXT_EXTENSIONS = {'txt'}


class DummyDataset(Dataset):
    FLAGS = ['img', 'txt', 'npy']

    def __init__(self, real_path, fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'img',
                 transform=None,
                 tokenizer=None,
                 mode="orig") -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            "Got unexpected modality flag: {} or {}".format(real_flag, fake_flag)
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_foler = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        # assert self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_foler[index]
        real_data = self._load_modality(real_path, self.real_flag, mode=self.mode)
        fake_data = self._load_modality(fake_path, self.fake_flag, mode="orig")

        sample = dict(real=real_data, fake=fake_data, name=os.path.basename(real_path))
        return sample

    def _load_modality(self, path, modality, mode="orig"):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        elif modality == 'npy':
            data = self._load_npy(path, mode=mode)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_npy(self, path, mode="orig"):
        data = np.load(path)
        if mode == 'stats':
            return data
        img = Image.fromarray(data)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True

    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()

    for batch_data in tqdm(dataloader):
        real = batch_data['real']
        if real_flag == 'txt':
            real_features = forward_modality(model, real, real_flag)
        else:
            real_features = batch_data['real']
            device = next(model.parameters()).device
            real_features = real_features.to(device)

        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]

    return score_acc / sample_num


@torch.no_grad()
def get_clip_features(dataloader, model, flag):
    features = []
    names = []
    for batch_data in tqdm(dataloader):
        data = batch_data['real']
        names.extend(batch_data['name'])
        feature = forward_modality(model, data, flag)
        feature = feature / feature.norm(dim=1, keepdim=True).to(torch.float32)
        features.append(feature)
    return torch.cat(features, dim=0), names


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


def clip_score(real_path, fake_path, clip_model='ViT-B/32', num_workers=None, batch_size=64):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    print('Loading CLIP model: {}'.format(clip_model))
    model, preprocess = clip.load(clip_model, device=device)

    dataset = DummyDataset(real_path, fake_path,
                           "npy", "npy",
                           transform=preprocess, tokenizer=clip.tokenize, mode="stats")
    dataloader = DataLoader(dataset, batch_size,
                            num_workers=num_workers, pin_memory=True)

    print('Calculating CLIP Score:')
    score = calculate_clip_score(dataloader, model,
                                 "npy", "img")
    print(f'{clip_model.replace("/", "-")} CLIP Score: {score:.4f}')
    return score


def clip_features(dataset_path, clip_model='ViT-B/32', num_workers=None, batch_size=64):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    print('Loading CLIP model: {}'.format(clip_model))
    model, preprocess = clip.load(clip_model, device=device)

    # find the extension of the dataset files
    extension = os.listdir(dataset_path)[0].split('.')[-1]
    if extension in IMAGE_EXTENSIONS:
        flag = 'img'
    elif extension in TEXT_EXTENSIONS:
        flag = 'txt'
    elif extension == 'npy':
        flag = 'npy'
    else:
        raise TypeError("Got unexpected extension: {}".format(extension))

    dataset = DummyDataset(dataset_path, dataset_path,
                           flag, flag,
                           transform=preprocess, tokenizer=clip.tokenize)

    dataloader = DataLoader(dataset, batch_size,
                            num_workers=num_workers, pin_memory=True)

    print('Calculating CLIP Features:')
    features, names = get_clip_features(dataloader, model, 'txt')
    features = features.cpu().numpy()
    save_path = os.path.join(os.path.dirname(dataset_path), f'{clip_model.replace("/", "-")}_clip_features')
    os.makedirs(save_path, exist_ok=True)

    for i, name in enumerate(names):
        np.save(os.path.join(save_path, f'{name[:-4]}.npy'), features[i])
    print('CLIP Features saved successfully!')
