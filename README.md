# Linear-Nonlinear-Units(LiNLU)
This repository is the official implementation of LiNLNet: Gauging Required Nonlinearity in Deep Neural Networks.
The defining feature of LiNLNets is the use of linear-nonlinear units (LiNLUs) as activation functions. The LiNLUs in a layer are given a single nonlinearity parameter $p$ confined to the range $0.5 - 1$; they gradually vary from ReLU-like functions to identity functions as parameter $p$ increases in the range.
The LiNLUs analyze the layer-wise optimal nonlinearity of LiNLNet and compress LiNLNets for the layers with $p^*=1$.

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

### Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 

[ImageNet](https://www.image-net.org/)

## Training
To train LiNL-Net on CIFAR-10 or ImageNet, run this command:
```train
cd LiNLU
python main.py --task <CIFAR-10 or ImageNet> --network <MLP or AlexNet or VGG16 or ResNet18> --mode train --activation LiNLU
```

## Evaluation
To evaluate LiNL-Net on CIFAR-10, or ImageNet, run this command:
```evaluation
cd LiNLU
python main.py --task <CIFAR-10 or ImageNet> --network <MLP or AlexNet or VGG16 or ResNet18> --mode eval --activation LiNLU
```


## Results
Our model achieves the following performance on: 

| CIFAR-10                                                            |
| Network       | Dataset        | Accuracy (%)   | # Linear layers   |
|---------------|----------------|----------------|-------------------|
| LiNL-MLP      | CIFAR-10       | 68.10 ± 0.17   | 1                 |
| MLP           | CIFAR-10       | 67.68 ± 0.15   | -                 |
| LiNL-MLP      | CIFAR-10       | 68.10 ± 0.17   | 1                 |
| MLP           | CIFAR-10       | 67.68 ± 0.15   | -                 |
| LiNL-MLP      | CIFAR-10       | 68.10 ± 0.17   | 1                 |
| MLP           | CIFAR-10       | 67.68 ± 0.15   | -                 |
| LiNL-MLP      | CIFAR-10       | 68.10 ± 0.17   | 1                 |
| MLP           | CIFAR-10       | 67.68 ± 0.15   | -                 |

*FMNISTnet : 32C5-P2-64C5-P2-600-10  
*CIFARnet : 64C5-128C5-P2-256C5-P2-512C5-256C5-1024-512-10

