#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "CustomNetworks.h"

using json = nlohmann::json;

struct TrainingParameters {
    double learning_rate;
    std::string optimizer_choice;
    std::string loss_function;
    int batch_size;
    int epochs;
    int num_samples;
    int num_classes;
};

struct InferenceParameters {
    int no_inferences;
};

class ConfigParser {
public:
    ConfigParser(const std::string& config_path);

    bool parseConfig();

    std::vector<NetworkLayer>& getNetworkLayers();
    json getInputShape();
    std::string getDevice();
    std::string getMode();
    TrainingParameters getTrainingParameters();
    InferenceParameters getInferenceParameters();
    std::string getTask();

private:
    std::string config_path_;
    json layers_;
    std::vector<NetworkLayer> network_layers_;
    json input_shape_;
    std::string device_;
    std::string mode_;
    json inference_params_;
    json training_params_;
    TrainingParameters training_parameters_;
    InferenceParameters inference_parameters_;
    std::string task_;
};