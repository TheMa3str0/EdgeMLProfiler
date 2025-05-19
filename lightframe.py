import argparse
import os # Added for path operations and directory creation
from datetime import datetime # Added for timestamp generation
from validate_config import validate_config
from logger import log_power
from profiler_runner import run_profiler_task
from power_analyzer import load_power_log, analyze_power_consumption

# In your main.py file:

def print_power_analysis_summary(analysis_results, profiler_name):
    print(f"\n--- Power Analysis Summary for {profiler_name} ---")

    if 'error' in analysis_results and analysis_results['error']:
        print(f"Note: {analysis_results['error']}")

    duration_s = analysis_results.get('total_duration_s', 'N/A')
    print(f"Profiled Duration: {duration_s:.3f} s" if isinstance(duration_s, float) else f"Profiled Duration: {duration_s}")

    num_ops = analysis_results.get('num_operations', 'N/A')
    op_unit = analysis_results.get('operation_unit', "operation")

    if isinstance(num_ops, (int, float)) and num_ops > 0:
        print(f"Number of Ops:     {num_ops:.0f} ({op_unit}s)")
    else:
        print(f"Number of Ops:     {num_ops} - Power per operation not meaningful.")

    avg_power = analysis_results.get('average_power_mw', 'N/A')
    min_power = analysis_results.get('min_power_mw', 'N/A')
    max_power = analysis_results.get('max_power_mw', 'N/A')

    print(f"Avg. Power Cons.:  {avg_power:.2f} mW" if isinstance(avg_power, float) else f"Avg. Power Cons.:  {avg_power}")
    print(f"Min. Power Cons.:  {min_power:.2f} mW" if isinstance(min_power, float) else f"Min. Power Cons.:  {min_power}")
    print(f"Max. Power Cons.:  {max_power:.2f} mW" if isinstance(max_power, float) else f"Max. Power Cons.:  {max_power}")

    print("-" * (len(f"--- Power Analysis Summary for {profiler_name} ---") -1))

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

    logs_dir = './logs'
    os.makedirs(logs_dir, exist_ok=True)


    config_file_name_base = os.path.splitext(os.path.basename(config_path))[0]
    mode = config_data['network']['mode']
    device = config_data['network']['device']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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
    python_log_filename = f"python_{config_file_name_base}_{mode}_{device}_{timestamp}_power_log.txt"
    python_power_log_path = os.path.join(logs_dir, python_log_filename)
    python_command_args = [python_executable, python_script_path, '-c', config_path]

    py_rc, py_start_time, py_end_time = run_profiler_task(
        profiler_name="Python (PyTorch)",
        executable_path_args=python_command_args,
        config_path_for_error_msg=config_path,
        power_log_file_path=python_power_log_path,
        power_logging_enabled=power_logging_enabled,
        logging_interval=logging_interval,
        log_power_func=log_power
    )

    if power_logging_enabled and py_start_time is not None and py_end_time is not None:
        print(f"\nAttempting power analysis for Python (PyTorch)...")
        py_power_data = load_power_log(python_power_log_path)
        if py_power_data:
            py_analysis_results = analyze_power_consumption(
                power_data=py_power_data,
                config_data=config_data,
                start_time_ns=py_start_time,
                end_time_ns=py_end_time
            )
            print_power_analysis_summary(py_analysis_results, "Python (PyTorch)")
        else:
            print(f"Could not load or empty power log for Python (PyTorch) from '{python_power_log_path}'. Skipping power analysis.")
    elif power_logging_enabled:
        print("\nSkipping Python (PyTorch) power analysis: Start/end times not captured, or power log was not created/is empty.")

    # --- C++ Profiler ---
    print("\n" + "="*60)
    cpp_executable_path = './cpp/build/profiler'
    cpp_log_filename = f"cpp_{config_file_name_base}_{mode}_{device}_{timestamp}_power_log.txt"
    cpp_power_log_path = os.path.join(logs_dir, cpp_log_filename)
    cpp_command_args = [cpp_executable_path, '-c', config_path]

    cpp_rc, cpp_start_time, cpp_end_time = run_profiler_task(
        profiler_name="C++ (LibTorch)",
        executable_path_args=cpp_command_args,
        config_path_for_error_msg=config_path,
        power_log_file_path=cpp_power_log_path,
        power_logging_enabled=power_logging_enabled,
        logging_interval=logging_interval,
        log_power_func=log_power
    )

    if power_logging_enabled and cpp_start_time is not None and cpp_end_time is not None:
        print(f"\nAttempting power analysis for C++ (LibTorch)...")
        cpp_power_data = load_power_log(cpp_power_log_path)
        if cpp_power_data:
            cpp_analysis_results = analyze_power_consumption(
                power_data=cpp_power_data,
                config_data=config_data,
                start_time_ns=cpp_start_time,
                end_time_ns=cpp_end_time
            )
            print_power_analysis_summary(cpp_analysis_results, "C++ (LibTorch)")
        else:
            print(f"Could not load or empty power log for C++ (LibTorch) from '{cpp_power_log_path}'. Skipping power analysis.")
    elif power_logging_enabled:
        print("\nSkipping C++ (LibTorch) power analysis: Start/end times not captured, or power log was not created/is empty.")

    print("\nAll profiling tasks complete.")

if __name__ == "__main__":
    main()