# Linear-Nonlinear-Units(LiNLU)
This repository is the official implementation of LiNLNet: Gauging Required Nonlinearity in Deep Neural Networks.
The defining feature of LiNLNets is the use of linear-nonlinear units (LiNLUs) as activation functions. The LiNLUs in a layer are given a single nonlinearity parameter $p$ confined to the range [0.5 - 1]; they gradually vary from ReLU-like functions to identity functions as parameter $p$ increases in the range.



## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

### Dataset
[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 

## Training
To train SNNs (FMNISTnet and CIFARnet) using BPLC on Fashion-MNIST or CIFAR-10, run this command:
```train
cd BPLC+NOSO
python main.py --task <FMNIST or CIFAR10> --network <FMNISTnet or CIFARnet> --mode train
```

## Evaluation
To evaluate SNNs(FMNISTnet and CIFARnet) on Fashion-MNIST or CIFAR-10, run this command:
```evaluation
cd BPLC+NOSO
python main.py --task <FMNIST or CIFAR10> --network <FMNISTnet or CIFARnet> --mode eval
```


## Results
Our model achieves the following performance on: 

| Method        | Network        | Dataset           | Accuracy (%) | # spikes (inference)  |
|---------------|----------------|-------------------|--------------|-----------------------|
| BPLC+NOSO     | FMNISTnet      | Fashion-MNIST     | 92.47%       | 14K ± 0.26K           |
| BPLC+NOSO     | CIFARnet       | CIFAR-10          | 89.77%       | 142K ± 1.86K          |

*FMNISTnet : 32C5-P2-64C5-P2-600-10  
*CIFARnet : 64C5-128C5-P2-256C5-P2-512C5-256C5-1024-512-10

