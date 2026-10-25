# ICCACA

Official training and evaluation code for the full ICCACA remote-sensing image change captioning model.

The public model has one fixed architecture:

`SegFormer-MiT-B1 -> IA-CDA -> CAKA -> Inception caption decoder`

Ablation switches and experiment-only model variants are intentionally not included.

## Installation

Python 3.8+ and a CUDA-enabled PyTorch installation are recommended.

```bash
pip install -r requirements.txt
```

METEOR evaluation also requires Java. Install `mmcv-full` with the wheel matching your PyTorch and CUDA versions if a prebuilt wheel is available.

## Data

The code supports `LEVIR_CC`, `Dubai_CC`, `WHU_CDC`, and `SECOND_CC_AUG` for training. Evaluation currently supports the first three datasets. Each dataset needs:

- an image directory;
- split files and `vocab.json` under a list directory;
- preprocessed caption token files.

The repository's `data/` directory contains the included split, vocabulary, and token metadata. Dataset images must be obtained separately according to their respective licenses.

## Training

```bash
python train.py \
  --data_name LEVIR_CC \
  --data_path /path/to/LEVIR_CC/images \
  --list_path ./data/LEVIR_CC/ \
  --token_folder ./data/LEVIR_CC/tokens/ \
  --max_length 41 \
  --train_batch_size 32 \
  --save_path ./checkpoints/LEVIR_CC
```

Use `python train.py --help` for optimizer, batch-size, and runtime options.
Training runs for 20 epochs by default; override this with `--num_epochs`.

## Evaluation

```bash
python test.py \
  --data_name LEVIR_CC \
  --data_folder /path/to/LEVIR_CC/images \
  --list_path ./data/LEVIR_CC/ \
  --token_folder ./data/LEVIR_CC/tokens/ \
  --max_length 41 \
  --checkpoint ./checkpoints/LEVIR_CC/model.pth
```

Both greedy decoding and beam search are supported. Run `python test.py --help` for details.
