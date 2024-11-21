#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "CustomNetworks.h"
#include "CustomDataset.h"

class TrainingProfiler {
public:
    TrainingProfiler(
        CustomNetwork& network,
        const std::string& device,
        const std::vector<int64_t>& input_shape,
        double learning_rate,
        const std::string& optimizer_choice,
        const std::string& loss_function,
        int batch_size,
        int epochs,
        int num_samples,
        int num_classes,
        const std::string& task)
        : network_(network), device_(device), input_shape_(input_shape),
          learning_rate_(learning_rate), optimizer_choice_(optimizer_choice),
          loss_function_(loss_function), batch_size_(batch_size), epochs_(epochs),
          num_samples_(num_samples), num_classes_(num_classes), task_(task) {}

    void train() {
        torch::optim::Optimizer* optimizer = nullptr;
        if (optimizer_choice_ == "adam") {
            optimizer = new torch::optim::Adam(network_->parameters(), torch::optim::AdamOptions(learning_rate_));
        } else if (optimizer_choice_ == "sgd") {
            optimizer = new torch::optim::SGD(network_->parameters(), torch::optim::SGDOptions(learning_rate_));
        } else if (optimizer_choice_ == "rmsprop") {
            optimizer = new torch::optim::RMSprop(network_->parameters(), torch::optim::RMSpropOptions(learning_rate_));
        } else {
            std::cerr << "Unsupported optimizer choice: " << optimizer_choice_ << std::endl;
            return;
        }

        // Define loss function
        torch::nn::CrossEntropyLoss criterion;
        torch::nn::MSELoss mse_criterion;

        auto [train_data, train_labels] = generate_mock_training_data(input_shape_, num_classes_, num_samples_, task_);

        CustomDataset custom_dataset;
        if (device_ == "gpu") {
            custom_dataset = CustomDataset(train_data.to(torch::kCUDA), train_labels.to(torch::kCUDA));
        } else {
            custom_dataset = CustomDataset(train_data, train_labels);
        }
        auto dataset = custom_dataset.map(torch::data::transforms::Stack<>());

        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size_).workers(2)
        );

        network_->train();

        long long training_duration = 0;
        if (device_ == "cpu") {
            std::cout << "Training on CPU..." << std::endl;
            network_->to(torch::kCPU);
            
            auto training_start = std::chrono::high_resolution_clock::now();
            for (int epoch = 0; epoch < epochs_; ++epoch) {
                for (auto& batch : *data_loader) {
                    auto input = batch.data;
                    auto label = batch.target;
                    optimizer->zero_grad();
                    torch::Tensor output = network_(input);
                    torch::Tensor loss;
                    if (loss_function_ == "categorical_crossentropy") {
                        loss = criterion(output, label);
                    } else if (loss_function_ == "mse") {
                        loss = mse_criterion(output, label);
                    } else {
                        std::cerr << "Unsupported loss function: " << loss_function_ << std::endl;
                        return;
                    }
                    loss.backward();
                    optimizer->step();
                }
            auto training_end = std::chrono::high_resolution_clock::now();
            training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start).count(); 
            }
        } else if (device_ == "gpu") {
            std::cout << "Training on GPU..." << std::endl;
            network_->to(torch::kCUDA);

            torch::cuda::synchronize();
            auto training_start = std::chrono::high_resolution_clock::now();
            for (int epoch = 0; epoch < epochs_; ++epoch) {
                for (auto& batch : *data_loader) {
                    auto input = batch.data;
                    auto label = batch.target;
                    optimizer->zero_grad();
                    torch::Tensor output = network_(input);
                    torch::Tensor loss;
                    if (loss_function_ == "categorical_crossentropy") {
                        loss = criterion(output, label);
                    } else if (loss_function_ == "mse") {
                        loss = mse_criterion(output, label);
                    } else {
                        std::cerr << "Unsupported loss function: " << loss_function_ << std::endl;
                        return;
                    }
                    loss.backward();
                    optimizer->step();
                }
                torch::cuda::synchronize();
                auto training_end = std::chrono::high_resolution_clock::now();
                training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start).count(); 
            }
        } else {
            std::cerr << "Error" << std::endl;
            return;
        }

        printMetrics(training_duration);
    }

private:
    void printMetrics(long long duration) {
        std::cout << "Trained for " << epochs_ << " epochs in "
                  << duration << " ms.\n";
        std::cout << "Time spent per epoch: " << static_cast<double>(duration) / epochs_
                  << " ms on average.\n";
    }

    std::tuple<torch::Tensor, torch::Tensor> generate_mock_training_data(const std::vector<int64_t>& input_shape, int64_t num_classes, int64_t num_samples, const std::string& task) {
        // Create random training data and labels
        //auto train_data = torch::randn({num_samples, input_shape[0], input_shape[1], input_shape[2]}, torch::kFloat32);
        std::vector<int64_t> data_shape = {num_samples};
        data_shape.insert(data_shape.end(), input_shape.begin(), input_shape.end());
        auto train_data = torch::randn(data_shape, torch::kFloat32);
        torch::Tensor train_labels;
        /* if (num_classes > 1) {
            // For classification, use torch::randint to generate integer labels
            train_labels = torch::randint(0, num_classes, {num_samples}, torch::kInt64);
        } else {
            // For regression, use torch::randn to generate continuous labels
            train_labels = torch::randn({num_samples}, torch::kFloat32);
        } */
        // Classification
        if (task == "classification") {
            train_labels = torch::randint(0, num_classes, {num_samples}, torch::kInt64);
        }
        // Multi-class Regression
        else if (task == "regression" && num_classes > 1) {
            train_labels = torch::randn({num_samples, num_classes}, torch::kFloat32);
        }
        // Single-output Regression
        else {
            train_labels = torch::randn({num_samples}, torch::kFloat32);
            train_labels = train_labels.view({-1, 1});
        }

        return std::make_tuple(train_data, train_labels);
    }

    CustomNetwork& network_;
    const std::vector<int64_t> input_shape_;
    const std::string device_;
    const double learning_rate_;
    const std::string& optimizer_choice_;
    const std::string& loss_function_;
    const int batch_size_;
    const int epochs_;
    const int num_samples_;
    const int num_classes_;
    const std::string& task_;
};