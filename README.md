# EdgeMLProfiler: ML Framework Comparison Tool

## Overview

**EdgeMLProfiler** is a user-friendly and lightweight tool designed for efficiently comparing inference and training speeds as well as power consumption across different machine learning frameworks. Currently supporting LibTorch and PyTorch, EdgeMLProfiler provides a straightforward way to assess the performance of these frameworks.

## Getting Started

### Requirements

Before using EdgeMLProfiler, ensure you have the following prerequisites installed:

- [PyTorch](https://pytorch.org/get-started/locally/) and [LibTorch](https://pytorch.org/get-started/locally/)
- [CMake](https://cmake.org/) (version 3.10 or higher)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (for GPU acceleration)
- [nlohmann/json.hpp](https://github.com/nlohmann/json) (JSON library)
- [jetson-stats](https://github.com/rbonghi/jetson_stats) (for power monitoring) 

### Installation

To get started with EdgeMLProfiler, follow these simple steps:

1. Clone the EdgeMLProfiler repository:
   ```bash
   git clone https://github.com/TheMa3str0/EdgeMLProfiler.git
   ```

2. **Navigate to the EdgeMLProfiler directory:**
    ```bash
    cd EdgeMLProfiler
    ```

3. **Build the C++ module using CMake:**
    - Navigate to the `/cpp` folder:
        ```bash
        cd cpp
        ```

    - Create a `build` directory for CMake:
        ```bash
        mkdir build
        cd build
        ```

    - Run CMake to configure the build:
        ```bash
        cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
        ```

    - Build the C++ part:
        ```bash
        make
        ```

4. **Return to the main directory:**
    ```bash
    cd ../..
    ```
### Running the tool

To run the EdgeMLProfiler, run the startup script with the location to the config file.

1. **Run EdgeMLProfiler:**
    ```bash
    python3 lightframe.py --config /path/to/config
    ```

## Configuration File Options Guide

The configuration file contains essential details about the network architecture and the operation you want to perform, whether it's training or inference.

### Network Configuration

1. **`device` (string, required):**
   - Options: `'cpu'`, `'gpu'`
   - Description: Specify the device to be used for computations.

2. **`input_shape` (list of integers, required):**
   - Example: `[width, height, channels]`
   - Description: Define the input shape for the neural network. Must be a list of positive integers representing the width, height, and number of channels.

3. **`mode` (string, required):**
   - Options: `'inference'`, `'training'`
   - Description: Specify whether the network is used for inference or training.

4. **`task` (string, required):**
   - Options: `'classification'`, `'regression'`
   - Description: Define the task type for the network.

5. **`power_measurement` (dictionary, required for both `'inference'` and `'training'` modes):**
   - Example: `{ 'status': 'on', 'logging_interval': 1 }`
   - Description: Specifies whether power measurement is enabled (`'status'`) and sets the interval in seconds for logging power data (`'logging_interval'`).

6. **`inference_params` (dictionary, required for `'inference'` mode):**
   - Example: `{ 'no_inferences': 5000 }`
   - Description: Additional parameters required for inference mode. Must include the number of inferences (`'no_inferences'`).

7. **`training_params` (dictionary, required for `'training'` mode):**
   - Example: `{ 'optimizer': 'adam', 'learning_rate': 0.001, 'loss_function': 'categorical_crossentropy', 'batch_size': 32, 'epochs': 10, 'num_samples': 1000 }`
   - Description: Additional parameters required for training mode, including optimizer type, learning rate, loss function, batch size, number of epochs, and number of training samples.

8. **`warmup_params` (dictionary, required for warmup before profiling):**
   - Example: `{ 'no_operations': 500 }`
   - Description: Specify number of operations for warmup.

### Layer Configuration

1. **`type` (string, required):**
   - Options: `'conv2d'`, `'dense'`, `'maxpool2d'`, `'flatten'`, `'batchnorm2d'`, `'averagepool2d'`, `'dropout'`, `'residual_block'`
   - Description: Specify the type of layer.

2. **`activation_function` (string):**
   - Options: `'relu'`, `'softmax'`, `'tanh'`
   - Description: Specify the activation function for the layer.

3. **`io_shape` (list of integers):**
   - Example: `[input_shape, output_shape]`
   - Description: Define the input/output shape for the layer. Must be a list of positive integers.

4. **`conv_params` (dictionary, required for `'conv2d'` layers):**
   - Example: `{ 'padding': 'same', 'kernel_size': [3, 3], 'stride': [1, 1] }`
   - Description: Parameters specific to convolutional layers, including padding, kernel size, and stride.

5. **`maxpool_params` (dictionary, required for `'maxpool2d'` layers):**
   - Example: `{ 'padding': 'same', 'kernel_size': [2, 2], 'stride': [2, 2] }`
   - Description: Parameters specific to max pooling layers, including padding, kernel size, and stride.

6. **`batchnorm_params` (dictionary, required for `'batchnorm2d'` layers):**
   - Example: `{ 'num_features': [64] }`
   - Description: Parameters specific to batch normalization layers, including the number of features.

7. **`dropout_params` (dictionary, required for `'dropout'` layers):**
   - Example: `{ 'p': 0.5, 'inplace': True }`
   - Description: Parameters specific to dropout layers, including the dropout probability (`'p'`) and whether it is in-place (`'inplace'`).

8. **`averagepool_params` (dictionary, required for `'averagepool2d'` layers):**
   - Example: `{ 'kernel_size': [2, 2], 'stride': [2, 2] }`
   - Description: Parameters specific to average pooling layers, including kernel size and stride.

9. **`residual_params` (dictionary, required for `'residual_block'` layers):**
   - Example: `{ 'in_channels': 64, 'out_channels': 128, 'stride': 1 }`
   - Description: Parameters specific to residual block layers, including input channels, output channels, and stride.

### Example Configurations

Explore examples like alexnet, vggnet19, and resnet34 in the configs/ folder to jumpstart your custom setups.

## Publication
[An Open-Source Tool for Analyzing the Time Efficiency of Machine Learning on Edge Devices](https://ieeexplore.ieee.org/abstract/document/10815819)

Enjoy using EdgeMLProfiler for efficient ML framework speed and power comparisons!
