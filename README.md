# APTP: Adaptive Prompt-Tailored Pruning of T2I Diffusion Models
An open-source implementation of the paper ["Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models"](https://openreview.net/forum?id=ekR510QsYF)

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
Here are the steps to prepare the data for training mentioned in the paper. You can also use your own data with some minor modifications to the code.

### 1. Download Conceptual Captions
Download Conceptual captions using the instructions provided [here](https://github.com/igorbrigadir/DownloadConceptualCaptions). Place the downloaded data in a directory of your choice. Keep the structure of the data as downloaded. The directory should look like this:
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
There are some urls in the Conceptual Captions dataset that are not valid. These could be removed to ensure a more efficient training process.

### 2. Download MS-COCO 2014
#### 2.1 Download the training and validation images
Download [2014 train](http://images.cocodataset.org/zips/train2014.zip) and [2014 val](http://images.cocodataset.org/zips/val2014.zip) images from the [COCO website](http://cocodataset.org/#download). Place the downloaded images in a directory of your choice. 

#### 2.2 Download the annotations
Download the [2014 train/val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) from the [COCO website](http://cocodataset.org/#download). Place the downloaded annotations in the same directory as the images.

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

To utilize this implementation, run the following command:

```bash
accelerate launch --multi_gpu scripts/prune.py \
  --train_data_dir "/path/to/data/dir" \
  --data_files "list/of/data/file/dirs"\
  --pruning_p 0.9 \
```

## Additional Notes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
