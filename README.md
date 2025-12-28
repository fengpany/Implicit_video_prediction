# Hierarchical Temporal-spatial Residual Model for Long-term Video Prediction

This is the official repo of article "Hierarchical Temporal-spatial Residual Model for Long-term Video Prediction".

## Abstract

Predicting long-term future frames in videos is challenging due to prediction ambiguities and error amplification over time. These issues become more pronounced in distant frames, where small errors accumulate into significant discrepancies. Few studies have addressed hierarchical spatial-temporal representations that effectively manage video stochasticity, particularly across datasets with varying spatial distributions. Given the importance of spatial distribution analysis in computer vision, hierarchical spatial modeling has shown to outperform many non-autoregressive likelihood-based models, particularly in video spatial analysis. In this work, we propose a Hierarchical Temporal-Spatial Residual Model for long-term video prediction, which captures the residual distribution between the prior and posterior, facilitating a richer representation of the stochastic features present in videos. Specifically, we propose a hierarchical residual generative model that improves the latent state space's ability to capture spatial features in videos. This approach enhances generalization across diverse spatial distributions in video data. By explicitly modeling the residual nature of the data and aligning the approximate posterior with the prior, our method better captures stochastic variations. As a result, it significantly boosts performance in long-term video prediction tasks. Evaluations on three challenging datasets show that our model outperforms both temporal model-based and convolutional neural network-based approaches. 

## Major contribution

* We propose a novel implicit hierarchical spatial reconstructed architecture to enhance hierarchical information transmission by modeling the residual distribution between prior and posterior, without excessively increasing computational cost.
* We introduce a temporal-spatial decomposing spatial reconstruction framework for video prediction task that learns the variable prior distributions in the spatial dimension through modeling the posterior distribution. 

## Installation

### Environment

* tensorflow==2.4.1
* tensorflow-probability==0.12.0
* python==3.8.18
* Nvidia GPU V100 32GB
* CUDA 11.8

### Setup

```
pip install -r requirement.txt
```

## Run

### Train

* For Moving Mnist

  ```
  python train.py --logdir ./logs --config ./configs/mmnist.yml
  ```

  For kth

  ```
  python train.py --logdir ./logs --config ./configs/kth.yml
  ```

  For GQN-Mazes

  ```
  python train.py --logdir ./logs --config ./configs/mazes.yml
  ```

  

### Test

* For Moving Mnist

  ```
  python eval_with_LPIPS --logdir ./logs/mmnist/model --eval-seq-len 300 --open-loop-ctx 36
  ```

  For kth

  ```
  python eval_with_LPIPS --logdir ./logs/kth/model --eval-seq-len 300 --open-loop-ctx 36
  ```

  For GQN-Mazes

  ```
  python eval_with_LPIPS --logdir ./logs/mazes/model --eval-seq-len 300 --open-loop-ctx 36
  ```

  

## Datasets

* Download link
  * Moving Mnist: https://archive.org/download/moving_mnist/moving_mnist_2digit.zip
  * GQN_Mazes: https://archive.org/download/gqn_mazes/gqn_mazes.zip
  * KTH: [Recognition of human actions](https://www.csc.kth.se/cvap/actions/)
    * For the Kth dataset, we split the dataset by assigning the actions of people numbered 1 through 20 to the training set, while the actions of people numbered 21 through 25 are used for testing.

The dataset should be organized as follows:

```
Moving Mnist/
├── test-seq1000
├── train-seq100
  ├── 1.avi
  ├── 2.avi
KTH/
├── test
├── train
GQN-Mazes/
├── test
├── train
```

The preprocess for dataset is in `data_loader.py`.



## Model

### Model configs

* Moving Mnist: `./configs/mmnist.yml`
* KTH: `./configs/kth.yml`
* GQN-Mazes:`./configs/mazes.yml`

You can modify the .yml file to setup model parameters. 

### Checkpoints

We provide checkpoint files  for all datasets in [checkpoints](https://drive.google.com/drive/folders/1Gje2F2His-pQGjNKTxzXiXM3bRrK-5c-). 

