import constructor
import inference
import training
import json
from parameter_parser import get_inference_params, get_training_params, get_warmup_params
import argparse

def read_config_file(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='../configs/network_config.json', help='Path to the configuration file')
args = parser.parse_args()
config_data = read_config_file(args.c)
    
network = constructor.build_custom_net(config_data['network']['layers'])
#constructor.print_network_architecture(network)
    
device = config_data['network']['device']
warmup_params = get_warmup_params(config_data)

if config_data['network']['mode'] == 'inference':
    inference_params = get_inference_params(config_data)
    inference_time, start_time, end_time = inference.profile_custom(network, device, *inference_params, *warmup_params)
    total_time = inference_time * pow(10, -6)
    print(f"[START TIME] {start_time} - [END TIME] {end_time}")
    print(f"Ran {config_data['network']['inference_params']['no_inferences']} inferences in {total_time} ms.")
    print(f"Time spent per inference: {str(total_time / config_data['network']['inference_params']['no_inferences'])} ms on average.")
elif config_data['network']['mode'] == 'training':
    training_params = get_training_params(config_data)
    training_time, start_time, end_time = training.train_network(network, device, *training_params, *warmup_params)
    total_time = training_time * pow(10, -6)
    print(f"[START TIME] {start_time} - [END TIME] {end_time}")
    print(f"Trained for {config_data['network']['training_params']['epochs']} epochs in {total_time} ms.")
    print(f"Time spent per epoch: {total_time / config_data['network']['training_params']['epochs']} ms on average.")
else:
    print("Invalid mode specified in the configuration file.")