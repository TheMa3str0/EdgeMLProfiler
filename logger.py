import time
import psutil
import threading
from jtop import jtop

# Modify log_power to accept the stop event
def log_power(pid, stop_event, log_file="power_log.txt", interval=0.5):
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Error: Process with PID {pid} not found at logger start.")
        return
    except psutil.AccessDenied:
         print(f"Error: Permission denied to access process PID {pid}.")
         return


    print(f"Logger thread started for PID {pid}. Logging to {log_file}")
    try:
        with jtop() as jetson, open(log_file, "w") as f:
            f.write("timestamp_ns,power_tot_mw\n")

            while not stop_event.is_set():
                if not psutil.pid_exists(pid):
                     print(f"Logger: Process {pid} disappeared.")
                     break

                try:
                    if not p.is_running():
                        print(f"Logger: Process {pid} is no longer running.")
                        break
                except psutil.NoSuchProcess:
                    print(f"Logger: Process {pid} disappeared (NoSuchProcess during is_running check).")
                    break

                if jetson.ok():
                    stats = jetson.stats
                    power = stats.get('Power TOT', 0)
                    timestamp = int(time.time() * 1e9)
                    f.write(f"{timestamp},{power}\n")
                    # Optionally flush buffer periodically if real-time view is needed
                    # f.flush()
                else:
                    print("Warning: jtop is not ok(). Skipping power reading.")

                stop_event.wait(timeout=interval)

    except Exception as e:
        print(f"Error during power logging for PID {pid}: {e}")
    finally:
        print(f"Power logging loop finished for PID {pid}. Power log saved to {log_file}")