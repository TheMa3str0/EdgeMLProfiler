#include "ConfigParser.h"
#include <fstream>
#include <iostream>
#include "CustomNetworks.h"

ConfigParser::ConfigParser(const std::string& config_path) : config_path_(config_path) {}

bool ConfigParser::parseConfig() {
    std::ifstream config_file(config_path_);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open the configuration file.\n";
        return false;
    }

    json config_data;
    config_file >> config_data;

    layers_ = config_data["network"]["layers"];
    input_shape_ = config_data["network"]["input_shape"];
    device_ = config_data["network"]["device"];
    mode_ = config_data["network"]["mode"];
    task_ = config_data["network"]["task"];

    if (mode_ == "inference") {
        inference_params_ = config_data["network"]["inference_params"];
    } else if (mode_ == "training") {
        training_params_ = config_data["network"]["training_params"];
    }

    return true;
}

std::vector<NetworkLayer>& ConfigParser::getNetworkLayers() {
    for (const auto& layer : layers_) {
        NetworkLayer layer_info;
        layer_info.type = layer["type"];

        // Check if "activation_function" exists before extracting it
        if (layer.contains("activation_function")) {
            layer_info.activation_function = layer["activation_function"];
        }

        // Check if "io_shape" exists before extracting it
        if (layer.contains("io_shape")) {
            const auto io_shape = layer["io_shape"];
            for (const auto& shape : io_shape) {
                layer_info.io_shape.push_back(shape);
            }
        }

        if (layer_info.type == "conv2d") {
            ConvParams convolution_parameters;
            convolution_parameters.kernel_size = layer["conv_params"]["kernel_size"].get<std::vector<int64_t>>();
            convolution_parameters.stride = layer["conv_params"]["stride"].get<std::vector<int64_t>>();
            if (layer["conv_params"]["padding"].is_number()) {
                convolution_parameters.padding = std::to_string(layer["conv_params"]["padding"].get<int64_t>());
            } else {
                convolution_parameters.padding = layer["conv_params"]["padding"].get<std::string>();
            }
            layer_info.conv_params = convolution_parameters; 
        } else if (layer_info.type == "maxpool2d") {
            PoolParams maxpool_parameters;
            maxpool_parameters.kernel_size = layer["maxpool_params"]["kernel_size"].get<std::vector<int64_t>>();
            maxpool_parameters.stride = layer["maxpool_params"]["stride"].get<std::vector<int64_t>>();
            if (layer["maxpool_params"]["padding"].is_number()) {
                maxpool_parameters.padding = std::to_string(layer["maxpool_params"]["padding"].get<int64_t>());
            } else {
                maxpool_parameters.padding = layer["maxpool_params"]["padding"].get<std::string>();
            }
            layer_info.pool_params = maxpool_parameters;
        } else if (layer_info.type == "batchnorm2d") {
            BatchnormParams batchnorm_parameters;
            batchnorm_parameters.num_features = layer["batchnorm_params"]["num_features"].get<std::vector<int64_t>>();
            layer_info.batchnorm_params = batchnorm_parameters;
        } else if (layer_info.type == "dropout") {
            DropoutParams dropout_parameters;
            dropout_parameters.probability = layer["dropout_params"]["p"];
            dropout_parameters.inplace = layer["dropout_params"]["inplace"];
            layer_info.dropout_params = dropout_parameters;
        } else if (layer_info.type == "averagepool2d") {
            PoolParams averagepool_parameters;
            averagepool_parameters.kernel_size = layer["averagepool_params"]["kernel_size"].get<std::vector<int64_t>>();
            averagepool_parameters.stride = layer["averagepool_params"]["stride"].get<std::vector<int64_t>>();
            if (layer["averagepool_params"]["padding"].is_number()) {
                averagepool_parameters.padding = std::to_string(layer["averagepool_params"]["padding"].get<int64_t>());
            } else {
                averagepool_parameters.padding = layer["averagepool_params"]["padding"].get<std::string>();
            }
            layer_info.pool_params = averagepool_parameters;
        } else if (layer_info.type == "residual_block") {
            ResidualParams residual_parameters;
            residual_parameters.in_channels = layer["residual_params"]["in_channels"];
            residual_parameters.out_channels = layer["residual_params"]["out_channels"];
            residual_parameters.stride = layer["residual_params"]["stride"];
            layer_info.residual_params = residual_parameters;
        }
        
        network_layers_.push_back(layer_info);
    }
    return network_layers_;
}

json ConfigParser::getInputShape() { return input_shape_; }
std::string ConfigParser::getDevice() { return device_; }
std::string ConfigParser::getMode() { return mode_; }

InferenceParameters ConfigParser::getInferenceParameters() {
    inference_parameters_.no_inferences = inference_params_["no_inferences"];
    return inference_parameters_; 
}

TrainingParameters ConfigParser::getTrainingParameters() {
    training_parameters_.learning_rate = training_params_["learning_rate"];
    training_parameters_.optimizer_choice = training_params_["optimizer"];
    training_parameters_.loss_function = training_params_["loss_function"];
    training_parameters_.batch_size = training_params_["batch_size"];
    training_parameters_.epochs = training_params_["epochs"];
    training_parameters_.num_samples = training_params_["num_samples"];
    training_parameters_.num_classes = network_layers_.back().io_shape[1];
    return training_parameters_;
}

std::string ConfigParser::getTask() { return task_; }
