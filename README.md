# APTP: Adaptive Prompt-Tailored Pruning of T2I Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2406.12042-red.svg)](https://arxiv.org/abs/2406.12042) 
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/rezashkv/APTP)


The implementation of the
paper ["Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models"]()

<p align="center">
  <img src="assets/fig_1.gif" alt="APTP Overview" width="600" />
</p>
<p align="left">
  <em>APTP: We prune a text-to-image diffusion model like Stable Diffusion (left) into a mixture of efficient experts (right) in a prompt-based manner. Our prompt router routes distinct types of prompts to different experts, allowing experts' architectures to be separately specialized by removing layers or channels.</em>
</p>

<p align="center">
  <img src="assets/fig_2.gif" alt="APTP Pruning Scheme" width="600" />
</p>
<p align="left">
  <em>APTP pruning scheme. We train the prompt router and the set of architecture codes to prune a
T2I diffusion model into a mixture of experts. The prompt router consists of three modules. We use
a Sentence Transformer as the prompt encoder to encode the input prompt into a representation z. Then,
the architecture predictor transforms z into the architecture embedding e that has the same dimensionality as
architecture codes. Finally, the router routes the embedding e into an architecture code a(i). We use optimal
transport to evenly distribute the prompts in a training batch among the architecture codes. The architecture code
a(i) = (u(i), v(i)) determines pruning the model’s width and depth. We train the prompt router’s parameters and
architecture codes in an end-to-end manner using the denoising objective of the pruned model L<sub>DDPM</sub>, distillation
loss between the pruned and original models L<sub>distill</sub>, average resource usage for the samples in the batch R, and
contrastive objective L<sub>cont</sub>, encouraging embeddings e preserving semantic similarity of the representations z.
</em>
</p>

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
   - [Download Conceptual Captions](#1-download-conceptual-captions)
   - [Download MS-COCO 2014](#2-download-ms-coco-2014)
3. [Training](#training)
   - [Pruning](#1-pruning)
   - [Data Preparation for Fine-tuning](#2-data-preparation-for-fine-tuning)
   - [Fine-tuning](#3-fine-tuning)
4. [Image Generation](#image-generation)
5. [Evaluation](#evaluation)
   - [FID Score](#1-fid-score)
   - [CLIP Score](#2-clip-score)
   - [CMMD](#3-cmmd)
6. [Baselines](#baselines)
7. [License](#license)
8. [Citation](#citation)

## Installation

Follow these steps to set up the project:

### 1. Create Conda Environment

Use the provided [env.yaml](env.yaml) file:

```bash
conda env create -f env.yaml
```

### 2. Activate the Conda Environment

Activate the environment:

```bash
conda activate pdm
```

### 3. Install Project Dependencies

From the project root directory, install the dependencies:

```bash
pip install -e .
```

## Data Preparation

Prepare the data for training as mentioned in the paper. You can also adapt aptp for your own dataset with minor code modifications.

### 1. Download Conceptual Captions

Follow the instructions [here](https://github.com/igorbrigadir/DownloadConceptualCaptions) to download Conceptual Captions. Place the data in a directory of your choice, maintaining the structure:

```
conceptual_captions
├── Train_GCC-training.tsv
├── Val_GCC-1.1.0-Validation.tsv
├── training
│   ├── 10007_560483514 
│   └── ...
└── validation
    ├── 1852290_2006010568
    └── ...
```

#### 1.1 Remove Corrupt Images (Optional)

There are some urls in the Conceptual Captions dataset that are not valid. The download will result in some corrupt files that can't be opened. These could be removed to ensure a more
efficient training.

### 2. Download MS-COCO 2014

#### 2.1 Download the training and validation images

Download [2014 train](http://images.cocodataset.org/zips/train2014.zip)
and [2014 val](http://images.cocodataset.org/zips/val2014.zip) images from
the [COCO website](http://cocodataset.org/#download). Place them in your chosen directory.

#### 2.2 Download the annotations

Download the [2014 train/val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
and place them in the same directory as the images. Your directory should look like this:

```
coco
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   └── ...
└── images
    ├── train2014
    │   ├── COCO_train2014_000000000009.jpg
    │   └── ...
    └── val2014
        ├── COCO_val2014_000000000042.jpg
        └── ...
```

## Training

Training is done in two stages: pruning the pretrained T2I model (Stable Diffusion 2.1 in this case) and fine-tuning each expert on the prompts assigned to it.
Configuration files for both Conceptual Captions and MS-COCO are provided in the [configs](configs) directory. 
You can use these configuration files to run the pruning process. Sample multi-node [SLURM](https://slurm.schedmd.com/) and [PBS](https://www.openpbs.org/) scripts can be found in [cluster scripts](cluster_scripts). 

### 1. Pruning

 You can use the following command to run pruning:

```bash

accelerate launch scripts/aptp/prune.py \
    --base_config_path path/to/configs/pruning/file.yaml \
    --cache_dir /path/to/.cache/huggingface/ \
    --wandb_run_name WANDB_PRUNING_RUN_NAME 
```
This creates a checkpoint directory named "wandb_run_name" in the logging directory specified in the config file. 

### 2. Data Preparation for Fine-tuning
The pruning stages results in $K$ architecture codes (experts). We need to assign each training prompt to its corresponding expert for fine-tuning. Assuming the pruning checkpoint directory is `pruning_checkpoint_dir`, you can use the following command to run this filtering process:

```bash
accelerate launch scripts/aptp/filter_dataset.py \
    --pruning_ckpt_dir path/to/pruning_checkpoint_dir \
    --base_config_path path/to/configs/filtering/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/
```
This creates two files named `{DATASET}_train_mapped_indices.pt` and `{DATASET}_validation_mapped_indices.pt` in the `pruning_checkpoint_dir` directory. These files contain the expert id for each prompt in the training and validation sets.


### 3. Fine-tuning
Fine-tune an expert on the prompts assigned to it:

```bash
accelerate launch scripts/aptp/finteune.py \
    --pruning_ckpt_dir path/to/pruning_checkpoint_dir \
    --expert_id INDEX \
    --base_config_path path/to/configs/finetuning/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/ \
    --wandb_run_name WANDB_FINETUNING_RUN_NAME
```

## Image Generation

Generate images using the experts:

```bash
accelerate launch scripts/metrics/generate_fid_images.py \
    --finetuning_ckpt_dir path/to/pruning_checkpoint_dir \
    --expert_id INDEX \
    --base_config_path path/to/configs/img_generation/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/
```
The generated images will be saved in the`{DATASET}_fid_images` directory within the root finetuning checkpoint directory.

## Evaluation
To evaluate APTP, we report the FID, CLIP Score, and CMMD.

### 1. FID Score
We use [clean-fid](https://github.com/GaParmar/clean-fid) to calculate the FID score. The numbers reported in the paper are calculated using this pytorch legacy mode.

#### 1.1 Conceptual Captions Preparation
We report FID on the validation set of Conceptual Captions. So we can use the same validation mapped indices file created in the filtering process. First, we need to resize the images to 256x256. You can use the [provided script](scripts/metrics/resize_and_save_images.py). It will save the images as numpy arrays in the same root directory of the dataset.


#### 1.2 MS-COCO Preparation
We sample 30k images from the 2014 validation set of MS-COCO. Check out the [sample and resize script](scripts/metrics/sample_coco_30k.py). We need a [filtering step](#2-data-preparation-for-fine-tuning) on this subset to assign prompts to experts.

#### 1.3 Generate Custom Statistics

Generate custom statistics for both sets of reference images::
```bash
from cleanfid import fid
fid.make_custom_stats(dataset, dataset_path, mode="legacy_pytorch") # mode can be clean too.
```

Now we can calculate the FID score for the generate images using [the provided script](scripts/metrics/fid.py).

### 2. CLIP Score
To calculate clip score, we use [this library](https://github.com/Taited/clip-score). Extract features of reference images with the [clip feature extraction script](scripts/metrics/clip_features.py) and calculate the score using the [clip score script](scripts/metrics/clip_score.py).

### 3. CMMD
We use the [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch) library to calculate CMMD. Refer to [save_refs.py](cmmd-pytorch/save_refs.py) and [compute_cmmd.py](cmmd-pytorch/compute_cmmd.py) scripts for reference set feature extraction and distance calculation details.


## Baselines
Refer to [here](configs/baselines) for config files for all baselines mentioned in the paper. The scripts to run these baselines are in [this directory](scripts/baselines).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this work useful, please consider citing the following paper:

```bibtex
@article{2024aptp,
  title={Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models},
  author={Shirkavand, Reza and Ganjdanesh, Alireza, and Gao, Shangqian and Huang, Heng},
  journal={arXiv preprint arXiv:2406.12042},
  year={2024}
}
```
