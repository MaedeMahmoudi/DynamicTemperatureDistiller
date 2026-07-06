# Dynamic Temperature Distiller

Knowledge Distillation with Dynamic Temperature for Deep Learning Models

## Description

Implementation of dynamic temperature-based knowledge distillation for training compact neural networks. This repository includes various distillation methods with a focus on adaptive temperature scheduling.

## Features

- Dynamic temperature adjustment during training
- Multiple distillation strategies:
  - Hinton's original KD
  - Attention-based distillation
  - Data-free distillation
  - Dynamic temperature distillation
- Support for MNIST dataset
- Easy to extend for custom datasets

## Project Structure
DynamicTemperatureDistiller/
├── distillation/
│ ├── baseDistiller.py
│ ├── hintonDistiller.py
│ ├── dynamicDistiller.py
│ ├── attentionDistiller.py
│ ├── datafreeDistiller.py
│ ├── datasetDistiller.py
│ ├── utils.py
│ └── requirements.txt
├── train_mnist.py
└── README.md


## Installation

```bash
git clone https://github.com/MaedeMahmoudi/DynamicTemperatureDistiller.git
cd DynamicTemperatureDistiller

python3 -m venv venv
source venv/bin/activate

pip install -r distillation/requirements.txt
‍‍Usage

Train with default settings:
bash

python train_mnist.py
گ
