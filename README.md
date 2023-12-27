# diffusion-pruning
An open-source implementation of diffusion model pruning.

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


## Usage

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
