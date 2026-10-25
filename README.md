# ICCACA

This repository is the official project page for ICCACA, a remote-sensing image change captioning model.

## Code Availability

The complete source code, configuration files, pretrained models, and detailed usage instructions will be made publicly available after the manuscript is formally accepted for publication.

## Acknowledgements

We sincerely thank the authors and contributors of DGAT and Chg2Cap, as well as the broader open-source community. Their excellent research and publicly available implementations provided valuable references and support for this work.

We will update the project links and citation information when the complete code is released.

## Installation

Python 3.8+ and a CUDA-enabled PyTorch installation are recommended.

```bash
pip install -r requirements.txt
```

METEOR evaluation requires Java. Install `mmcv-full` with the wheel matching your PyTorch and CUDA versions.

## Data Preparation

Training supports `LEVIR_CC`, `Dubai_CC`, `WHU_CDC`, and `SECOND_CC_AUG`. Evaluation currently supports `LEVIR_CC`, `Dubai_CC`, and `WHU_CDC`.

Each dataset requires:

- an image directory;
- split files and `vocab.json` in the list directory;
- preprocessed caption token files.

The `data/` directory contains split, vocabulary, and token metadata. Dataset images must be obtained separately under their respective licenses.

## Training

The following example trains the full model on LEVIR-CC with a training batch size of 32 and validation/test batch size of 1:

```bash
python train.py \
  --data_name LEVIR_CC \
  --data_path /path/to/LEVIR_CC/images \
  --list_path ./data/LEVIR_CC/ \
  --token_folder ./data/LEVIR_CC/tokens/ \
  --vocab_file vocab \
  --max_length 41 \
  --allow_unknown 1 \
  --train_batch_size 32 \
  --valid_batch_size 1 \
  --num_workers 8 \
  --num_epochs 20 \
  --gpu_id 0 \
  --save_path ./checkpoints/LEVIR_CC
```

Training runs for 20 epochs by default. Run `python train.py --help` for all available runtime and optimization options.

## Evaluation

Greedy decoding:

```bash
python test.py \
  --data_name LEVIR_CC \
  --data_folder /path/to/LEVIR_CC/images \
  --list_path ./data/LEVIR_CC/ \
  --token_folder ./data/LEVIR_CC/tokens/ \
  --vocab_file vocab \
  --max_length 41 \
  --allow_unknown 1 \
  --checkpoint /path/to/model.pth \
  --gpu_id 0 \
  --workers 8 \
  --test_batchsize 1 \
  --decode_method greedy
```

Beam search:

```bash
python test.py \
  --data_name LEVIR_CC \
  --data_folder /path/to/LEVIR_CC/images \
  --list_path ./data/LEVIR_CC/ \
  --token_folder ./data/LEVIR_CC/tokens/ \
  --vocab_file vocab \
  --max_length 41 \
  --checkpoint /path/to/model.pth \
  --gpu_id 0 \
  --workers 8 \
  --test_batchsize 1 \
  --decode_method beam \
  --beam_size 3 \
  --length_penalty 0.6
```

Run `python test.py --help` for all evaluation options.
