# PSDN

PSDN is a two-stage progressive denoising framework for unsupervised visible-infrared person re-identification (US-VI-ReID). The ALS module is designed for global soft label correction, while the SMS module performs local boundary refinement. The two modules work together to suppress pseudo-label noise and boost cross-modal feature alignment performance.

The task of this paper is unsupervised visible-infrared person re-identification (US-VI-ReID). Given visible and infrared pedestrian images, the model achieves cross-modal pedestrian identity matching and retrieval without any manual annotations. The proposed PSDN framework adopts TransReID ViT as the feature extraction backbone. It learns modality-invariant discriminative features via global label smoothing and local boundary refinement, alleviates pseudo-label noise caused by clustering, and improves the stability of cross-modal feature alignment as well as retrieval accuracy.

## overview

The framework of our work is illustrated as follows:
[<img width="4252" height="2250" alt="framework" src="https://github.com/1097714058-coder/PSDN/issues/1#issue-4870728698" />]
The framework contains two main modules:
1. **ALS: Aptive Label Smooth**

   ALS computes adaptive label smoothing weights for each sample based on cross-cluster distribution, cluster distance and category frequency. It then transforms traditional one-hot hard pseudo-labels into dynamic probability soft labels, allowing the training pipeline to mitigate large-scale clustering mismatches and eliminate global pseudo-label noise.

2. **SMS: Soft Margin Smooth**

   SMS refines the smoothed soft labels produced by ALS using cross-modal neighbor consistency. It combines neighbor consistency voting, difference-aware exponential weighting and local adaptive decision boundary constraint to suppress residual noisy samples and stabilize model training gradients.


# Prepare Datasets
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare\_sysu.py and prepare\_regdb.py to prepare the training data (convert to market1501 format).

# Prepare Pre-trained model
We adopt the self-supervised pre-trained models (ViT-B/16+ICS) from [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file).
Download link:https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view

# Training

We utilize 2 4090 GPUs for training.

**examples:**

SYSU-MM01:

1. Train:
```shell
sh train_sysu.sh
```


2. Test:
```shell
sh test_sysu.sh
```

RegDB:

1. Train:
:
```shell
sh train_regdb.sh
```

2. Test:
```shell
sh test_regdb.sh
```
## Repository Structure

```text
PSDN/
+-- datasets/                  # Dataset loading, preprocessing, and samplers
+-- model/
|   +-- build.py               # Main model definition and forward process
|   +-- clip_model.py          # CLIP image/text encoders
|   +-- CrossEmbeddingLayer_tse.py
|   +-- objectives.py          # TAL, CGR, and CMM losses
+-- processor/
|   +-- processor.py           # Training, inference, and VNM sample division
+-- solver/                    # Optimizer and learning-rate scheduler
+-- utils/                     # Options, logging, metrics, and utility functions
+-- train.py                   # Training entry point
+-- test.py                    # Evaluation entry point
+-- rank.py                    # Text-query Top-K visualization
+-- run_vecm.sh                # Example training script
```



## Supported Datasets

The code supports common unsupervised visible-infrare person re-identification datasets:

```text
SYSU-MM01
RegDB
LLCM
```

Set `--root_dir` to the dataset root path. The corresponding dataset reader should provide image paths, text descriptions, and identity labels.

During training, noisy image-text correspondences can be injected or loaded using `--noisy_rate` and `--noisy_file`.


## Evaluation

Before evaluation, edit the `sub` variable in `test.py` and set it to the training output directory, for example:

```text
run_logs/RSTPReid/202xxxxx_VECM_TAL+...
```

Then run:

```bash
python test.py
```

The evaluator reports three retrieval settings:

```text
BGE      # global image-text features
TSE      # local enhanced features
BGE+TSE  # fused global and local similarities
```

Metrics:

```text
Rank-1
Rank-5
Rank-10
mAP
mINP
```


## Main Options

### Basic Training Options

```text
--pretrain_choice      CLIP pretrained model, default: ViT-B/16
--img_size             Input image size, default in code: (384, 128)
--stride_size          ViT patch stride, default: 16
--text_length          Maximum text token length, default: 77
--batch_size           Training batch size, default: 64
--test_batch_size      Evaluation batch size, default: 512
--select_ratio         TSE token selection ratio, default: 0.3
--tau                  TAL temperature, default: 0.015
--margin               TAL margin, default: 0.1
```


