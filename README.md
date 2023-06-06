# ActionNet
This repository contains code and documentation for training a ResNet model on a human action recognition dataset. The goal is to accurately classify different human actions such as eating, sitting, calling, and more. The experiment includes hyperparameter search, train-validation-test split, and the use of transfer learning.

#### Show your appreciation: If you find this project useful, please consider showing your support by starring it on GitHub. Your support means a lot! :star:

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Link-green.svg)](https://www.shamimahamed.com/actionnet)

#### View experiments on [W&B](https://wandb.ai/shamim/ActionNet)
[![Open in Wandb](https://img.shields.io/badge/Open%20in-Wandb-blue)](https://wandb.ai/shamim/ActionNet)

![](Images/wandb.jpg?raw=true)

### Download Data
Check `src/Data/README.md` file for instruction on how to setup dataset

### Setup W&B api key
``` CLI
export WANDB_API_KEY=<your_key>
echo $WANDB_API_KEY
```

### Training
``` CLI
python train.py --n_epochs 10 --batch_size 128 --learning_rate 0.001
```
View train.py file for more detals


# License
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)



