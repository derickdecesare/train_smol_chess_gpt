import os
import time
import psutil
import sys
from datetime import datetime

def get_process_info():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'python' in proc.info['name'] and 'prepare.py' in str(proc.info['cmdline']):
            return proc
    return None

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def main():
    print("\033[2J\033[H", end='')  # Clear screen
    print(f"=== Data Preparation Progress Monitor ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")

    # Check prepare.py process
    proc = get_process_info()
    if not proc:
        print("prepare.py process not found!")
        sys.exit(1)

    # Check binary files
    files = ['train.bin', 'val.bin']
    for fname in files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"{fname}: {format_size(size)}")
        else:
            print(f"{fname}: Not created yet")

    # Check process resources
    try:
        cpu_percent = proc.cpu_percent(interval=1)
        mem_info = proc.memory_info()
        print(f"\nProcess Info (PID {proc.pid}):")
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(f"Memory Usage: {format_size(mem_info.rss)}")

        # Check log file
        if os.path.exists('preparation.log'):
            with open('preparation.log', 'r') as f:
                lines = f.readlines()
                print("\nLast log entries:")
                for line in lines[-5:]:
                    print(line.strip())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("Process ended or access denied")
        sys.exit(1)

if __name__ == '__main__':
    main()
