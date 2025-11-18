# Hierarchical Temporal-spatial Residual Model for Long-term Video Prediction

## Installation

```
pip install -r requirement.txt
```

## Run

* Test:

```
python eval_with_LPIPS.py
```

* Train:

```
python train.py
```



## Datasets

* Download link
  * Moving Mnist: https://archive.org/download/moving_mnist/moving_mnist_2digit.zip
  * GQN_Mazes: https://archive.org/download/gqn_mazes/gqn_mazes.zip
  * KTH: [Recognition of human actions](https://www.csc.kth.se/cvap/actions/)

The dataset should be organized as follows:

```
Moving Mnist/
├── test/
├── train/
```

The preprocess for dataset is in `data_loader.py`
 


## Checkpoints

We provide checkpoint files  for all datasets in [checkpoints](https://drive.google.com/drive/folders/1Gje2F2His-pQGjNKTxzXiXM3bRrK-5c-). 

