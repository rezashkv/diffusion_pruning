# diffusion-pruning
An open-source implementation of diffusion model pruning.

## Usage

To utilize this implementation, run the following command:

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu scripts/prune.py \
  --train_data_dir "/path/to/data/dir" \
  --data_files "list/of/data/file/dirs"\
  --pruning_p 0.9 \
