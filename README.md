Zero-Shot Video Moment Retrieval with Angular Reconstructive Text Embeddings
=====
PyTorch implementation of .
This is our anonymous implementation codes of "Zero-Shot Video Moment Retrieval with Angular Reconstructive Text Embeddings" (ART).
## 1. Environment
This repository is implemented based on [PyTorch](http://pytorch.org/) with Anaconda.</br>

### Get code and environment
Make your own environment (If you use docker envronment, you just clone the code and execute it.)
```bash
conda create --name ART --file requirements.txt
conda activate ART
```

### Working environment
RTX 3090

Ubuntu 20.04.1

## 2. Prepare data
We employ the pre-trained I3D model to extract the Charades-STA features, while C3D models extract the ActivityNet-Caption.
You can download the video, text CLIP features and pretrained model at the google drive:
[Charades](https://drive.google.com/drive/folders/1bJuOrB3sWhQNyAm4GhzI9SQPxs0-wkNT)

And you should put the features in the data/. folder

#### 3. Evaluating pre-trained models
* Using **anaconda** environment
```bash
# Evaluate ART model trained from Charades-STA Dataset
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
                     --config config/charades/config.yml \
                     --checkpoint pretrained_model/charades/best.pkl \
                     --method tgn_lgi \
                     --dataset charades
```

train the charades dataset

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train \
                     --config_path config/charades/config.yml \
                     --method_type tgn_lgi \
                     --dataset charades \
                     --tag base --seed 2049
```

The pre-trained models will report following scores.
While re-implementing this code, the reproduced numbers are slightly different.


Dataset              | R@0.3 | R@0.5 | R@0.7 | mIoU
-------------------- | ------| ------| ------| ------
Charades-STA         | 57.31 | 41.13 | 22.12 | 39.01
