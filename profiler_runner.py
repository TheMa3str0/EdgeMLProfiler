import subprocess
import threading

def run_profiler_task(
    profiler_name,
    executable_path_args, # e.g., ['python3', './path/to/script.py', '-c', 'config.json']
    config_path_for_error_msg,
    power_log_file_path,
    power_logging_enabled,
    logging_interval,
    log_power_func
):
    print(f"\n--- Starting {profiler_name} Profiling ---")
    if isinstance(executable_path_args, list):
        print(f"Command: {' '.join(executable_path_args)}")
    else:
        print(f"Executable: {executable_path_args}")

    proc = None
    log_thread = None
    stop_logging_event = None
    profiler_return_code = -1

    try:
        proc = subprocess.Popen(
            executable_path_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Started {profiler_name} profiler with PID: {proc.pid}")

        if power_logging_enabled:
            stop_logging_event = threading.Event()
            log_thread = threading.Thread(
                target=log_power_func,
                args=(proc.pid, stop_logging_event, power_log_file_path, logging_interval),
                daemon=True
            )
            print(f"Starting power logging for {profiler_name} (log file: {power_log_file_path})...")
            log_thread.start()
        else:
            print(f"Skipping power logging for {profiler_name} as per configuration.")

        print(f"Waiting for {profiler_name} (PID: {proc.pid}) to complete and gather its output...")

        stdout_bytes, stderr_bytes = proc.communicate()
        profiler_return_code = proc.returncode

        print(f"\n--- {profiler_name} Profiler Output ---")
        if stdout_bytes:
            stdout_str = stdout_bytes.decode('utf-8', errors='replace').strip()
            print(stdout_str)
        if stderr_bytes:
            stderr_str = stderr_bytes.decode('utf-8', errors='replace').strip()
            if stderr_str:
                print(f"\n--- {profiler_name} Profiler Error Output ---")
                print(stderr_str)
                print("-" * (len(f"--- {profiler_name} Profiler Error Output ---") -1))

        print(f"\n{profiler_name} profiler (PID: {proc.pid}) finished with return code: {profiler_return_code}")
        if profiler_return_code != 0 and profiler_name.startswith("C++"):
            print(f"Warning: {profiler_name} profiler exited with non-zero status {profiler_return_code}")


        if power_logging_enabled and log_thread is not None:
            print(f"Signaling {profiler_name} power logger to stop...")
            if stop_logging_event:
                stop_logging_event.set()
            log_thread.join(timeout=10)
            if log_thread.is_alive():
                print(f"Warning: {profiler_name} power logging thread did not exit cleanly within the timeout.")
            else:
                print(f"{profiler_name} power logging thread finished.")
        elif power_logging_enabled and log_thread is None:
            print(f"Warning: {profiler_name} power logging was flagged as enabled, but the logging thread was not initialized.")
        
        return profiler_return_code

    except FileNotFoundError:
        exec_name = executable_path_args[0] if executable_path_args else "Unknown Executable"
        print(f"Error: {profiler_name} profiler executable not found at '{exec_name}' (related to config: {config_path_for_error_msg})")
        print(f"Skipping {profiler_name} profiling.")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred while running the {profiler_name} profiler: {e}")
        print(f"Skipping {profiler_name} profiling.")
        if proc and proc.poll() is not None:
            profiler_return_code = proc.returncode
        return profiler_return_code
    finally:
        print(f"\nMain program execution for {profiler_name} profiler finished.")