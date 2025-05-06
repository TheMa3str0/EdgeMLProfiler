import argparse
from validate_config import validate_config
from logger import log_power
from profiler_runner import run_profiler_task

def main():
    parser = argparse.ArgumentParser(description='Lightframe: ML Framework Speed Comparison Tool')
    parser.add_argument('--config', type=str, default='./configs/network_config.json',
                        help='Path to the configuration file')
    args = parser.parse_args()
    config_path = args.config

    print(f'Using configuration file: {config_path}')

    is_valid, config_data = validate_config(config_path)
    if not is_valid:
        print("Exiting due to invalid config file.")
        exit(1)

    power_logging_enabled = False
    logging_interval = 1.0

    try:
        power_config = config_data['network']['power_measurement']
        if power_config['status'] == 'on':
            power_logging_enabled = True
            logging_interval = power_config['logging_interval']
            print(f"Power measurement status from config: ON (Interval: {logging_interval}s)")
        else:
            print("Power measurement status from config: OFF")
    except KeyError as e:
        print(f"Warning: Key '{e}' missing in 'power_measurement' config. Assuming power logging OFF.")
        power_logging_enabled = False


    # --- Python Profiler ---
    python_executable = 'python3'
    python_script_path = './python/profiler.py'
    python_power_log_path = './python_power_log.txt'
    python_command_args = [python_executable, python_script_path, '-c', config_path]

    run_profiler_task(
        profiler_name="Python (PyTorch)",
        executable_path_args=python_command_args,
        config_path_for_error_msg=config_path,
        power_log_file_path=python_power_log_path,
        power_logging_enabled=power_logging_enabled,
        logging_interval=logging_interval,
        log_power_func=log_power
    )

    # --- C++ Profiler ---
    print("\n" + "="*60)
    cpp_executable_path = './cpp/build/profiler'
    cpp_power_log_path = './cpp_power_log.txt'
    cpp_command_args = [cpp_executable_path, '-c', config_path]

    run_profiler_task(
        profiler_name="C++ (LibTorch)",
        executable_path_args=cpp_command_args,
        config_path_for_error_msg=config_path,
        power_log_file_path=cpp_power_log_path,
        power_logging_enabled=power_logging_enabled,
        logging_interval=logging_interval,
        log_power_func=log_power
    )

    print("\nAll profiling tasks complete.")

if __name__ == "__main__":
    main()