import matplotlib.pyplot as plt
import numpy as np

# Data provided directly
results = {
    'PyTorch': {
        'lenet': {'CPU': 0.495, 'GPU': 0.166},
        'alexnet': {'CPU': 8.636, 'GPU': 0.486},
        'vgnet19': {'CPU': 61.824, 'GPU': 3.048},
        'resnet': {'CPU': 15.719, 'GPU': 1.627}
    },
    'LibTorch': {
        'lenet': {'CPU': 0.5966, 'GPU': 0.123},
        'alexnet': {'CPU': 8.5162, 'GPU': 0.503},
        'vgnet19': {'CPU': 62.395, 'GPU': 3.078},
        'resnet': {'CPU': 14.54, 'GPU': 1.173}
    }
}

# Plotting
networks = list(results['PyTorch'].keys())
frameworks = list(results.keys())
bar_width = 0.2
index = np.arange(len(networks))  # Use numpy array for index

plt.figure(figsize=(12, 6))

for i, network in enumerate(networks):
    for j, framework in enumerate(frameworks):
        cpu_time = results[framework][network]['CPU']
        gpu_time = results[framework][network]['GPU']
        
        # Check if any value is too large
        if max(cpu_time, gpu_time) > 1000:
            plt.subplot(2, 1, 1)  # First subplot for first half of the data
            plt.bar(index + j * bar_width, [cpu_time], bar_width, label=f'{framework} + CPU')  # Positioning for CPU bar
            plt.bar(index + (j + 1) * bar_width, [gpu_time], bar_width, label=f'{framework} + GPU')  # Positioning for GPU bar
            plt.xticks(index + 0.5 * bar_width * len(frameworks), networks)
            plt.title(f'Inference Time Comparison (First Half)')
            plt.legend()

            plt.subplot(2, 1, 2)  # Second subplot for second half of the data
            plt.bar(index + j * bar_width, [cpu_time], bar_width, label=f'{framework} + CPU')  # Positioning for CPU bar
            plt.bar(index + (j + 1) * bar_width, [gpu_time], bar_width, label=f'{framework} + GPU')  # Positioning for GPU bar
            plt.xticks(index + 0.5 * bar_width * len(frameworks), networks)
            plt.title(f'Inference Time Comparison (Second Half)')
            plt.legend()
        else:
            plt.bar(index + j * bar_width, [cpu_time], bar_width, label=f'{framework} + CPU')  # Positioning for CPU bar
            plt.bar(index + (j + 1) * bar_width, [gpu_time], bar_width, label=f'{framework} + GPU')  # Positioning for GPU bar

plt.xlabel('Networks')
plt.ylabel('Inference Time (ms)')
plt.title(f'Inference Time Comparison')
plt.xticks(index + 0.5 * bar_width * len(frameworks), networks)
plt.legend()
plt.tight_layout()
plt.savefig(f'all_networks.png')  # Save the plot as an image
plt.show()
