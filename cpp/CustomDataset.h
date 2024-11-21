#pragma once
#include <torch/torch.h>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
public:
    // Default constructor
    CustomDataset() {}

    // Constructor that takes two tensors
    CustomDataset(torch::Tensor inputs1, torch::Tensor inputs2) : inputs1(inputs1), inputs2(inputs2) {}

    // Override the size method to return the number of samples in the dataset
    torch::data::Example<> get(size_t index) override {
        torch::Tensor input1 = inputs1[index];
        torch::Tensor label = inputs2[index];

        // Return a single input tensor and label
        return {input1, label};
    }

    // Override the size method to return the number of samples in the dataset
    torch::optional<size_t> size() const override {
        return inputs1.size(0);
    }

private:
    torch::Tensor inputs1;
    torch::Tensor inputs2;
};
