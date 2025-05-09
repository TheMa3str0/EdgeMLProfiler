# power_analyzer.py (library version - strictly mW focus)

import csv
# import json # Not strictly needed here if config_data is just accessed as dict
import bisect
import sys 

def load_power_log(power_log_file_path):
    power_data = []
    try:
        with open(power_log_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Skip header
            except StopIteration: # Empty file
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

def get_interpolated_power(target_ns, power_data_sorted_ns):
    if not power_data_sorted_ns:
        print("Warning (power_analyzer): Power data is empty for interpolation. Returning 0 mW.", file=sys.stderr)
        return 0.0 

    timestamps_ns = [p[0] for p in power_data_sorted_ns]
    powers_mw = [p[1] for p in power_data_sorted_ns]

    if target_ns <= timestamps_ns[0]:
        if target_ns < timestamps_ns[0]: 
            print(f"Warning (power_analyzer): target_ns {target_ns} is before first log point {timestamps_ns[0]}. Using power at first point ({powers_mw[0]} mW).", file=sys.stderr)
        return powers_mw[0]
    if target_ns >= timestamps_ns[-1]:
        if target_ns > timestamps_ns[-1]: 
            print(f"Warning (power_analyzer): target_ns {target_ns} is after last log point {timestamps_ns[-1]}. Using power at last point ({powers_mw[-1]} mW).", file=sys.stderr)
        return powers_mw[-1]

    idx_after = bisect.bisect_left(timestamps_ns, target_ns)
    
    ts_before_val = timestamps_ns[idx_after - 1]
    p_before_val = powers_mw[idx_after - 1]
    ts_after_val = timestamps_ns[idx_after]
    p_after_val = powers_mw[idx_after]

    if ts_after_val == ts_before_val: 
        return p_after_val 
    
    interpolated_power_mw = p_before_val + (p_after_val - p_before_val) * \
                            (float(target_ns - ts_before_val)) / (float(ts_after_val - ts_before_val))
    return interpolated_power_mw


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
        results['error'] = f"Time interval [{start_time_ns}, {end_time_ns}] is entirely outside power log range [{min_log_ts}, {max_log_ts}]."
        print(f"Error (power_analyzer): {results['error']}", file=sys.stderr)
        return results

    integration_points = []
    p_start_mw = get_interpolated_power(start_time_ns, power_data)
    integration_points.append((start_time_ns, p_start_mw))

    for ts_ns, p_mw in power_data:
        if start_time_ns < ts_ns < end_time_ns:
            integration_points.append((ts_ns, p_mw))

    p_end_mw = get_interpolated_power(end_time_ns, power_data)
    if end_time_ns > integration_points[-1][0]:
         integration_points.append((end_time_ns, p_end_mw))
    elif end_time_ns == integration_points[-1][0] and end_time_ns != start_time_ns: 
        integration_points[-1] = (end_time_ns, p_end_mw)

    unique_points_dict = {ts: p_val for ts, p_val in integration_points}
    final_integration_points = sorted(unique_points_dict.items())
    
    total_duration_s = (end_time_ns - start_time_ns) / 1_000_000_000.0
    results['total_duration_s'] = total_duration_s
    
    # Internal energy calculation for average power
    total_energy_mj_internal = 0.0 
    avg_power_mw_calculated = 0.0

    if len(final_integration_points) < 2:
        print("Warning (power_analyzer): Less than two distinct points for integration. Average power may be imprecise.", file=sys.stderr)
        if final_integration_points:
            avg_power_mw_calculated = final_integration_points[0][1]
        # total_energy_mj_internal remains 0 or could be calculated if needed, but not primary
    else:
        for i in range(len(final_integration_points) - 1):
            ts1_ns, p1_mw = final_integration_points[i]
            ts2_ns, p2_mw = final_integration_points[i+1]
            if ts2_ns == ts1_ns : continue
            avg_power_interval_mw = (p1_mw + p2_mw) / 2.0
            delta_t_s = (ts2_ns - ts1_ns) / 1_000_000_000.0
            total_energy_mj_internal += avg_power_interval_mw * delta_t_s
        
        if total_duration_s > 0:
            avg_power_mw_calculated = total_energy_mj_internal / total_duration_s
        elif total_energy_mj_internal > 0 : 
            avg_power_mw_calculated = float('inf')
        elif final_integration_points : # duration 0, energy 0
             avg_power_mw_calculated = final_integration_points[0][1]
    
    results['average_power_mw'] = avg_power_mw_calculated

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
            else:
                steps_per_epoch = float(tp['num_samples']) / batch_size
                num_operations = steps_per_epoch * float(tp['epochs'])
            operation_unit = "training step"
        else:
            print(f"Warning (power_analyzer): Unknown mode '{mode}' in config. Cannot determine ops.", file=sys.stderr)
        
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