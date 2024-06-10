# APTP: Adaptive Prompt-Tailored Pruning of T2I Diffusion Models

An implementation of the
paper ["Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models"](https://openreview.net/forum?id=ekR510QsYF)

## Installation

To get started, follow these steps:

### 1. Create Conda Environment

Use the provided [env.yaml](env.yaml) file to create a Conda environment:

```bash
conda env create -f env.yaml
```

### 2. Activate the Conda Environment

Activate the newly created Conda environment:

```bash
conda activate pdm
```

### 3. Install Project Dependencies

Navigate to the project root directory and install the project source using pip:

```bash
pip install -e .
```

## Data Preparation

Here are the steps to prepare the data for training mentioned in the paper. You can also use your own data with some
minor modifications to the code.

### 1. Download Conceptual Captions

Download Conceptual captions using the instructions
provided [here](https://github.com/igorbrigadir/DownloadConceptualCaptions). Place the downloaded data in a directory of
your choice. Keep the structure of the data as downloaded. The directory should look like this:

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

#### 1.1 (Optional) Remove Corrupt Images

There are some urls in the Conceptual Captions dataset that are not valid. These could be removed to ensure a more
efficient training process.

### 2. Download MS-COCO 2014

#### 2.1 Download the training and validation images

Download [2014 train](http://images.cocodataset.org/zips/train2014.zip)
and [2014 val](http://images.cocodataset.org/zips/val2014.zip) images from
the [COCO website](http://cocodataset.org/#download). Place the downloaded images in a directory of your choice.

#### 2.2 Download the annotations

Download the [2014 train/val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) from
the [COCO website](http://cocodataset.org/#download). Place the downloaded annotations in the same directory as the
images.

The directory should look like this:

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
You can use these configuration files to run the pruning process. Sample [SLURM](https://slurm.schedmd.com/) scripts can be found in the [slurm_scripts](slurm_scripts) for all stages.

### 1. Pruning

 You can use the following command to run pruning:

```bash

accelerate launch scripts/aptp/prune.py \
    --base_config_path path/to/configs/pruning/file.yaml \
    --cache_dir /path/to/.cache/huggingface/ \
    --wandb_run_name WANDB_PRUNING_RUN_NAME 
```
This will create a checkpoint directory named "wandb_run_name" in the logging directory specified in the config file. 

### 2. Data Preparation for Fine-tuning
The pruning stages results in $K$ architecture codes (experts). We need to assign training prompts to its corresponding expert for fine-tuning. Assuming the pruning checkpoint directory is `pruning_checkpoint_dir`, you can run the following command to run this filtering process:

```bash
accelerate launch scripts/aptp/filter_dataset.py \
    --pruning_ckpt_dir path/to/pruning_checkpoint_dir \
    --base_config_path path/to/configs/filtering/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/
```
This process will create two files named `{DATASET}_train_mapped_indices.pt` and `{DATASET}_validation_mapped_indices.pt` in the `pruning_checkpoint_dir` directory. These files contain the expert ids for each prompt in the training and validation sets.
### 3. Fine-tuning
After filtering the dataset, you can use the following command to fine-tune an expert on the prompts assigned to it:

```bash
accelerate launch scripts/aptp/finteune.py \
    --pruning_ckpt_dir path/to/pruning_checkpoint_dir \
    --expert_id INDEX \
    --base_config_path path/to/configs/finetuning/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/ \
    --wandb_run_name WANDB_FINETUNING_RUN_NAME
```

## Image Generation

To generate images using the experts run:

```bash
accelerate launch scripts/metrics/generate_fid_images.py \
    --finetuning_ckpt_dir path/to/pruning_checkpoint_dir \
    --expert_id INDEX \
    --base_config_path path/to/configs/img_generation/dataset.yaml \
    --cache_dir /path/to/.cache/huggingface/
```
It will save the generated images in the parent fine-tuning checkpoint directory that includes each expert's directory. The image directory will be named `{DATASET}_fid_images`

## Evaluation
To evaluate APTP, we report the FID score of the generated images, as well as CLIP Score and CMMD.

### 1. FID Score


```bash

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
