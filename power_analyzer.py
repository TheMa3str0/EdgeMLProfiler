import csv
import sys
import os

def trim_power_log_file(power_log_file_path, start_time_ns, end_time_ns):
    if not os.path.exists(power_log_file_path):
        return

    scoped_rows = []

    try:
        with open(power_log_file_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            try:
                header_row = next(reader)
                scoped_rows.append(header_row)
            except StopIteration:
                return

            for row_str_list in reader:
                if len(row_str_list) == 2:
                    try:
                        timestamp_ns_val = int(row_str_list[0])
                        if start_time_ns <= timestamp_ns_val <= end_time_ns:
                            scoped_rows.append(row_str_list)
                    except ValueError:
                        pass

    except IOError:
        return
    except Exception:
        return

    temp_file_path = power_log_file_path + ".tmp"
    try:
        with open(temp_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(scoped_rows)

        os.replace(temp_file_path, power_log_file_path)
    except (IOError, OSError) as e:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
            
    except Exception:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

def load_power_log(power_log_file_path):
    power_data = []
    try:
        with open(power_log_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"Info (power_analyzer): Power log file {power_log_file_path} is empty or contains only a header.", file=sys.stderr)
                return []

            expected_header = ['timestamp_ns', 'power_tot_mw']
            if header != expected_header:
                 print(f"Warning (power_analyzer): Power log header is '{header}', expected '{expected_header}'. "
                       "Proceeding by assuming format is <timestamp_nanoseconds>,<power_milliwatts>.", file=sys.stderr)

            for i, row in enumerate(reader):
                if len(row) != 2:
                    print(f"Warning (power_analyzer): Skipping malformed row {i+2} (file line number) in {power_log_file_path}: {row}", file=sys.stderr)
                    continue
                try:
                    timestamp_ns = int(row[0])
                    power_mw = float(row[1])
                    power_data.append((timestamp_ns, power_mw))
                except ValueError:
                    print(f"Warning (power_analyzer): Skipping row {i+2} (file line number) with invalid numeric data in {power_log_file_path}: {row}", file=sys.stderr)
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

    if start_time_ns >= end_time_ns:
        results['error'] = f"Invalid profiler time interval (start_time {start_time_ns} >= end_time {end_time_ns})."
        results['average_power_mw'] = 'N/A'
        results['min_power_mw'] = 'N/A'
        results['max_power_mw'] = 'N/A'
        results['total_duration_s'] = 'N/A'
        results['num_operations'] = 'N/A'
        results['operation_unit'] = 'N/A'
        print(f"Error (power_analyzer): {results['error']}", file=sys.stderr)
        return results

    results['total_duration_s'] = (end_time_ns - start_time_ns) / 1_000_000_000.0

    if not power_data:
        results['error'] = "No power data points found within the profiler's time interval."
        results['average_power_mw'] = 'N/A'
        results['min_power_mw'] = 'N/A'
        results['max_power_mw'] = 'N/A'
    else:
        power_values_in_scope = [p_mw for ts_ns, p_mw in power_data]
        
        results['average_power_mw'] = sum(power_values_in_scope) / len(power_values_in_scope)
        results['min_power_mw'] = min(power_values_in_scope)
        results['max_power_mw'] = max(power_values_in_scope)

    num_operations = 0.0
    operation_unit = "operation"
    results['num_operations'] = 'N/A'
    results['operation_unit'] = 'N/A'
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

        if mode in ['inference', 'training']:
             results['num_operations'] = num_operations
             results['operation_unit'] = operation_unit

    except KeyError as e:
        print(f"Warning (power_analyzer): Missing key '{e}' in config_data. Cannot determine ops.", file=sys.stderr)
    except Exception as e:
        print(f"Warning (power_analyzer): Error processing ops count from config: {e}", file=sys.stderr)

    return results