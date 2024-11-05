import os
import time
import psutil
from datetime import datetime, timedelta

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def get_prepare_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and any('prepare.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def main():
    data_dir = "data/lichess_hf_dataset"
    start_time = datetime.now()
    last_size = 0
    last_check = start_time

    print(f"=== Light Preparation Monitor Started at {start_time} ===")
    print(f"Monitoring directory: {data_dir}")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            now = datetime.now()
            elapsed = now - start_time
            time_since_last = now - last_check

            # Check prepare.py process
            proc = get_prepare_process()
            if not proc:
                print("\nPreparation process not found. Exiting...")
                break

            # Check binary files
            train_path = os.path.join(data_dir, "train.bin")
            val_path = os.path.join(data_dir, "val.bin")

            current_size = 0
            if os.path.exists(train_path):
                current_size += os.path.getsize(train_path)
            if os.path.exists(val_path):
                current_size += os.path.getsize(val_path)

            # Calculate progress metrics
            if time_since_last.total_seconds() >= 60:
                size_delta = current_size - last_size
                speed = size_delta / time_since_last.total_seconds()

                print(f"\n=== Status Update ({now.strftime('%H:%M:%S')}) ===")
                print(f"Elapsed time: {str(elapsed).split('.')[0]}")
                print(f"Process CPU: {proc.cpu_percent()}%")
                print(f"Process Memory: {format_size(proc.memory_info().rss)}")

                if os.path.exists(train_path):
                    print(f"train.bin: {format_size(os.path.getsize(train_path))}")
                if os.path.exists(val_path):
                    print(f"val.bin: {format_size(os.path.getsize(val_path))}")

                if size_delta > 0:
                    print(f"Processing speed: {format_size(speed)}/s")

                print("-" * 50)

                last_size = current_size
                last_check = now

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main()
