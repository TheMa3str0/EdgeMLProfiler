#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

struct ConvParams {
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;
    std::string padding;
};

struct PoolParams {
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> stride;
    std::string padding;
};

struct BatchnormParams {
    std::vector<int64_t> num_features;
};

struct DropoutParams {
    double probability;
    bool inplace;
};

struct ResidualParams {
    int in_channels;
    int out_channels;
    int stride;
};

struct NetworkLayer {
    std::string type;
    std::string activation_function;
    std::vector<int64_t> io_shape;
    ConvParams conv_params; // Only used if type is conv2d
    PoolParams pool_params; // Only used if type is maxpool2d or averagepool2d
    BatchnormParams batchnorm_params; // Only used if type is batchnorm2d
    DropoutParams dropout_params; // Only used if type is dropout
    ResidualParams residual_params; // Only used if type is residual_block;
};

struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride=1) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)));

        if (stride != 1) {
            shortcut = register_module("shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor residual = x;

        x = torch::relu(conv1->forward(x));
        x = conv2->forward(x);

        if (shortcut) {
            residual = shortcut->forward(residual);
        }

        x += residual;
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
};
TORCH_MODULE(ResidualBlock);

struct CustomNetworkImpl : torch::nn::Module {
    CustomNetworkImpl(const std::vector<NetworkLayer>& network_layers) {
        layers = register_module("layers", torch::nn::Sequential());
        for (const auto& layer_info : network_layers) {
            if (layer_info.type == "conv2d") {
                torch::nn::Conv2dOptions conv_options(layer_info.io_shape[0], layer_info.io_shape[1], layer_info.conv_params.kernel_size);
                conv_options.stride(layer_info.conv_params.stride);

                if (layer_info.conv_params.padding == "same") {
                    conv_options.padding(torch::kSame);
                } else {
                    int64_t numeric_padding = std::stoi(layer_info.conv_params.padding);
                    conv_options.padding(numeric_padding);
                }

                layers->push_back(torch::nn::Conv2d(conv_options));
            } else if (layer_info.type == "dense") {
                layers->push_back(torch::nn::Linear(
                    layer_info.io_shape[0], layer_info.io_shape[1]
                ));
            } else if (layer_info.type == "flatten") {
                layers->push_back(torch::nn::Flatten());
            } else if (layer_info.type == "maxpool2d") {
                torch::nn::MaxPool2dOptions maxpool_options(layer_info.pool_params.kernel_size);
                maxpool_options.stride(layer_info.pool_params.stride);
    
                if (layer_info.pool_params.padding == "same") {
                    // TODO SAME PADDING DOES NOT WORK IN LIBTORCH
                    std::cout << "NOT IMPLEMENTED YET" << std::endl;
                } else {
                    int64_t numeric_padding = std::stoi(layer_info.pool_params.padding);
                    maxpool_options.padding(numeric_padding);
                }

                layers->push_back(torch::nn::MaxPool2d(maxpool_options));
            } else if (layer_info.type == "batchnorm2d") {
                layers->push_back(torch::nn::BatchNorm2d(layer_info.batchnorm_params.num_features[0]));
            } else if (layer_info.type == "dropout") {
                layers->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(layer_info.dropout_params.probability).inplace(layer_info.dropout_params.inplace)));
            } else if (layer_info.type == "averagepool2d") {
                torch::nn::AvgPool2dOptions averagepool_options(layer_info.pool_params.kernel_size);
                averagepool_options.stride(layer_info.pool_params.stride);
    
                if (layer_info.pool_params.padding == "same") {
                    // TODO SAME PADDING DOES NOT WORK IN LIBTORCH
                    std::cout << "NOT IMPLEMENTED YET" << std::endl;
                } else {
                    int64_t numeric_padding = std::stoi(layer_info.pool_params.padding);
                    averagepool_options.padding(numeric_padding);
                }

                layers->push_back(torch::nn::AvgPool2d(averagepool_options));
            } else if (layer_info.type == "residual_block") {
                layers->push_back(ResidualBlock(layer_info.residual_params.in_channels,
                                                layer_info.residual_params.out_channels,
                                                layer_info.residual_params.stride));
            }

            if (layer_info.activation_function == "relu") {
                layers->push_back(torch::nn::ReLU());
            } else if (layer_info.activation_function == "softmax") {
                layers->push_back(torch::nn::Softmax(1));
            } else if (layer_info.activation_function == "tanh") {
                layers->push_back(torch::nn::Tanh());
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }

    torch::nn::Sequential layers;
};
TORCH_MODULE(CustomNetwork);