#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "CustomNetworks.h"

class InferenceProfiler {
public:
    InferenceProfiler(CustomNetwork& network, const std::string& device, const std::vector<int64_t>& input_shape, int no_inferences)
        : network_(network), device_(device), input_shape_(input_shape), no_inferences_(no_inferences) {}

    void profile() {
        torch::NoGradGuard no_grad;  // Disable gradient computation
        network_->eval();  // Set the network to evaluation mode
        std::vector<torch::Tensor> inputs;
        for (int i = 0; i < no_inferences_; ++i) {
            torch::Tensor input = torch::randn(input_shape_).unsqueeze(0);  // Add a batch dimension of size 1
            inputs.push_back(input);
        }
        if (device_ == "cpu") {
            profileOnCPU(inputs);
        } else if (device_ == "gpu") {
            profileOnGPU(inputs);
        } else {
            std::cerr << "Error: Invalid device specified.\n";
        }
    }

private:
    void profileOnCPU(const std::vector<torch::Tensor>& inputs) {
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& input : inputs) {
            torch::Tensor output = network_(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        long long start_seconds = long(std::chrono::duration<double>(start.time_since_epoch()).count() * pow(10, 9));
        long long end_seconds = long(std::chrono::duration<double>(end.time_since_epoch()).count() * pow(10, 9));
        printMetrics(duration, start_seconds, end_seconds);
    }

    void profileOnGPU(std::vector<torch::Tensor>& inputs) {
        torch::Device device(torch::kCUDA);  // Select the CUDA device
        network_->to(device);  // Move the network to GPU

        for (auto& input : inputs) {
            input = input.to(device);  // Move the individual tensor to the device
        }

        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& input : inputs) {
            torch::Tensor output = network_(input);
        }
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        long long start_seconds = long(std::chrono::duration<double>(start.time_since_epoch()).count() * pow(10, 9));
        long long end_seconds = long(std::chrono::duration<double>(end.time_since_epoch()).count() * pow(10, 9));
        printMetrics(duration, start_seconds, end_seconds);
    }

    void printMetrics(long long duration, long long start_time, long long end_time) {
        std::cout << "[START TIME] " << start_time << " - [END TIME]" << end_time << "\n";
        std::cout << "Ran " << no_inferences_ << " inferences in " << duration << " ms.\n";
        std::cout << "Time spent per inference: " << static_cast<double>(duration) / no_inferences_ << " ms on average.\n"; //Maybe set precision here?
    }

    CustomNetwork network_;
    std::string device_;
    std::vector<int64_t> input_shape_;
    int no_inferences_;
};