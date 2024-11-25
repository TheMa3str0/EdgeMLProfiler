import torch
import torch.nn as nn
import time
import json
import random

def profile_custom(network, device, input_shape, no_inferences, no_operations_warmup):
    torch.set_grad_enabled(False)
    network.eval()
    
    inputs = []
    for i in range(no_inferences):
        input = torch.randn(input_shape).unsqueeze(0)  # Add a batch dimension of size 1
        inputs.append(input)

    if device == 'cpu':
        if (no_operations_warmup > 0):
            print("Warming up...")
            for i in range(no_operations_warmup):
                output = network(random.choice(inputs))
        
        print("Running inferences on CPU...")
        start = time.time_ns()
        for input in inputs:
            output = network(input)
        return (time.time_ns() - start)
    elif device == 'gpu':
        device = torch.device('cuda')  # Select the CUDA device
        net = network.to(device)  # Move the network to GPU
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(device)  # Move the individual tensor to the device
        
        if (no_operations_warmup > 0):
            print("Warming up...")    
            for i in range(no_operations_warmup):
                output = network(random.choice(inputs))
        
        print("Running inferences on GPU...")
        torch.cuda.synchronize()
        start = time.time_ns()
        for input in inputs:
            output = net(input)
        torch.cuda.synchronize()
        return (time.time_ns() - start)
    else:
        print("Error")
        return 0
