import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def generate_mock_training_data(input_shape, num_classes, num_samples, task):
    # Create random training data and labels
    train_data = np.random.rand(num_samples, *input_shape).astype(np.float32)
    train_data = torch.tensor(train_data)
    
    # Classification
    if task == 'classification':
        train_labels = np.random.randint(0, num_classes, num_samples)
    # Multi-class Regression
    elif task == 'regression' and num_classes > 1:
        train_labels = np.random.randn(num_samples, num_classes).astype(np.float32)
    # Single-output Regression
    else:
        train_labels = np.random.randn(num_samples).astype(np.float32)
        train_labels = train_labels.reshape(-1, 1)
    train_labels = torch.tensor(train_labels)
    
    return train_data, train_labels

def train_network(network, device, optimizer_choice, learning_rate, loss_function, batch_size, epochs, num_samples, num_classes, input_shape, task, no_operations_warmup):
    if loss_function == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        print('Unsupported loss function:', loss_function)
        exit()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=learning_rate)
    else:
        print('Unsupported optimizer choice:', optimizer_choice)
        exit()

    # Generate mock training data and labels
    train_data, train_labels = generate_mock_training_data(input_shape, num_classes, num_samples, task)

    if device == 'gpu':
        train_dataset = torch.utils.data.TensorDataset(train_data.to('cuda'), train_labels.to('cuda'))
    else:
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    network.train()
    if device == 'cpu':
        network.to('cpu')
        if (no_operations_warmup > 0):
            print("Warming up...")
            for i in range(no_operations_warmup):
                with torch.no_grad():
                    inputs, labels = next(iter(train_loader))
                    outputs = network(inputs)
        
        print("Training on CPU...")
        start_time = time.time_ns()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        end_time = time.time_ns()
    elif device == 'gpu':
        network.to('cuda')
        if (no_operations_warmup > 0):
            print("Warming up...")
            for i in range(no_operations_warmup):
                with torch.no_grad():
                    inputs, labels = next(iter(train_loader))
                    outputs = network(inputs)  
        
        print("Training on GPU...")
        torch.cuda.synchronize()
        start_time = time.time_ns()
        for epoch in range(epochs):    
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 
        torch.cuda.synchronize()
        end_time = time.time_ns()
    else:
        print("Error")
        return 0

    return end_time - start_time

