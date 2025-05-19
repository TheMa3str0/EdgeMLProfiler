import csv
import sys
# bisect is no longer needed

def load_power_log(power_log_file_path):
    power_data = []
    try:
        with open(power_log_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"Warning (power_analyzer): Power log file {power_log_file_path} is empty.", file=sys.stderr)
                return []

            expected_header = ['timestamp_ns', 'power_tot_mw']
            if header != expected_header:
                 print(f"Warning (power_analyzer): Power log header is '{header}', expected '{expected_header}'. "
                       "Proceeding by assuming format is <timestamp_nanoseconds>,<power_milliwatts>.", file=sys.stderr)

            for i, row in enumerate(reader):
                if len(row) != 2:
                    print(f"Warning (power_analyzer): Skipping malformed row {i+2} in {power_log_file_path}: {row}", file=sys.stderr)
                    continue
                try:
                    timestamp_ns = int(row[0])
                    power_mw = float(row[1])
                    power_data.append((timestamp_ns, power_mw))
                except ValueError:
                    print(f"Warning (power_analyzer): Skipping row {i+2} with invalid numeric data in {power_log_file_path}: {row}", file=sys.stderr)
                    continue

        power_data.sort(key=lambda x: x[0])
        return power_data
    except FileNotFoundError:
        print(f"Error (power_analyzer): Power log file not found at {power_log_file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error (power_analyzer): Loading power log file {power_log_file_path}: {e}", file=sys.stderr)
        return []

def analyze_power_consumption(power_data, config_data, start_time_ns, end_time_ns):
    results = {}
    if not power_data:
        results['error'] = "No power data provided to analyze_power_consumption."
        print(f"Error (power_analyzer): {results['error']}", file=sys.stderr)
        return results

    if start_time_ns >= end_time_ns:
        results['error'] = f"Invalid time interval (start_time {start_time_ns} >= end_time {end_time_ns})."
        print(f"Error (power_analyzer): {results['error']}", file=sys.stderr)
        return results

    min_log_ts, max_log_ts = power_data[0][0], power_data[-1][0]
    if end_time_ns < min_log_ts or start_time_ns > max_log_ts:
        results['error'] = (f"Profiler interval [{start_time_ns}, {end_time_ns}] "
                            f"is entirely outside power log range [{min_log_ts}, {max_log_ts}]. "
                            "No data points to analyze for this interval.")
        print(f"Warning (power_analyzer): {results['error']}", file=sys.stderr)
        results['average_power_mw'] = 'N/A'
        results['min_power_mw'] = 'N/A'
        results['max_power_mw'] = 'N/A'
        results['total_duration_s'] = (end_time_ns - start_time_ns) / 1_000_000_000.0
    else:
        scoped_power_data = []
        for ts_ns, p_mw in power_data:
            if start_time_ns <= ts_ns <= end_time_ns:
                scoped_power_data.append((ts_ns, p_mw))

        if not scoped_power_data:
            print(f"Warning (power_analyzer): No power log entries found within the profiler's "
                  f"time interval [{start_time_ns}, {end_time_ns}]. "
                  f"Min/Max/Avg power will be 'N/A'.", file=sys.stderr)
            results['average_power_mw'] = 'N/A'
            results['min_power_mw'] = 'N/A'
            results['max_power_mw'] = 'N/A'
        else:
            power_values_in_scope = [p_mw for ts_ns, p_mw in scoped_power_data]
            results['average_power_mw'] = sum(power_values_in_scope) / len(power_values_in_scope)
            results['min_power_mw'] = min(power_values_in_scope)
            results['max_power_mw'] = max(power_values_in_scope)

    results['total_duration_s'] = (end_time_ns - start_time_ns) / 1_000_000_000.0

    num_operations = 0.0
    operation_unit = "operation"
    try:
        net_config = config_data['network']
        mode = net_config['mode']
        if mode == 'inference':
            num_operations = float(net_config['inference_params']['no_inferences'])
            operation_unit = "inference"
        elif mode == 'training':
            tp = net_config['training_params']
            batch_size = float(tp['batch_size'])
            if batch_size == 0:
                num_operations = 0
                print(f"Warning (power_analyzer): Batch size is 0. Number of training steps is 0.", file=sys.stderr)
            else:
                steps_per_epoch = float(tp['num_samples']) / batch_size
                num_operations = steps_per_epoch * float(tp['epochs'])
            operation_unit = "training step"
        else:
            print(f"Warning (power_analyzer): Unknown mode '{mode}' in config. Cannot determine ops.", file=sys.stderr)
            results['num_operations'] = 'N/A'
            results['operation_unit'] = 'N/A'

        if 'num_operations' not in results:
            results['num_operations'] = num_operations
            results['operation_unit'] = operation_unit

    except KeyError as e:
        print(f"Warning (power_analyzer): Missing key '{e}' in config_data. Cannot determine ops.", file=sys.stderr)
        results['num_operations'] = 'N/A'
        results['operation_unit'] = 'N/A'
    except Exception as e:
        print(f"Warning (power_analyzer): Error processing ops count from config: {e}", file=sys.stderr)
        results['num_operations'] = 'N/A'
        results['operation_unit'] = 'N/A'

    return results