import argparse
import subprocess
from validate_config import validate_config

def main():
    parser = argparse.ArgumentParser(description='Lightframe: ML Framework Speed Comparison Tool')
    parser.add_argument('--config', type=str, default='./configs/network_config.json', help='Path to the configuration file')
    args = parser.parse_args()
    config_path = args.config
    print(f'Using configuration file: {config_path}')
    
    if not validate_config(config_path):
        print("Exiting due to invalid config file.")
        exit()
        
    python_program_path = './python/profiler.py'
    print("\nPyTorch Results: ")
    subprocess.run(['python3', python_program_path, '-c', config_path])
        
    cpp_program_path = './cpp/build/profiler'
    print("\nLibTorch Results: ")
    subprocess.run([cpp_program_path, '-c', config_path])

if __name__ == "__main__":
    main()
