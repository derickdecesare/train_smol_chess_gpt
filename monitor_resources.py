import psutil
import time
import os
from datetime import datetime
import json

class ResourceMonitor:
    def __init__(self):
        self.history = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'timestamps': []
        }
        self.prepare_pid = None
        self.find_prepare_process()

    def find_prepare_process(self):
        """Find the prepare.py process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'] and any('prepare.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                    self.prepare_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def get_prepare_stats(self):
        """Get statistics for prepare.py process"""
        if not self.prepare_pid:
            return None
        try:
            proc = psutil.Process(self.prepare_pid)
            return {
                'cpu_percent': proc.cpu_percent(),
                'memory_percent': proc.memory_percent(),
                'num_threads': proc.num_threads(),
                'status': proc.status()
            }
        except psutil.NoSuchProcess:
            self.prepare_pid = None
            return None

    def monitor_resources(self):
        """Monitor system resources"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        prepare_stats = self.get_prepare_stats()

        timestamp = datetime.now().isoformat()
        self.history['timestamps'].append(timestamp)
        self.history['cpu_usage'].append(cpu_percent)
        self.history['memory_usage'].append(memory.percent)
        self.history['disk_io'].append({
            'read_bytes': disk.read_bytes,
            'write_bytes': disk.write_bytes
        })

        # Keep only last 60 data points (10 minutes of history)
        if len(self.history['timestamps']) > 60:
            for key in self.history:
                self.history[key] = self.history[key][-60:]

        report = [
            "=== System Resource Monitor ===",
            f"Time: {timestamp}",
            "",
            "System Resources:",
            f"CPU Usage: {cpu_percent}%",
            f"Memory Usage: {memory.percent}%",
            f"Available Memory: {memory.available / (1024**3):.1f} GB",
            f"Disk Read: {disk.read_bytes / (1024**2):.1f} MB",
            f"Disk Write: {disk.write_bytes / (1024**2):.1f} MB",
        ]

        if prepare_stats:
            report.extend([
                "",
                "prepare.py Process Stats:",
                f"CPU Usage: {prepare_stats['cpu_percent']}%",
                f"Memory Usage: {prepare_stats['memory_percent']:.1f}%",
                f"Threads: {prepare_stats['num_threads']}",
                f"Status: {prepare_stats['status']}"
            ])

        # Calculate rates
        if len(self.history['timestamps']) > 1:
            time_diff = (datetime.fromisoformat(self.history['timestamps'][-1]) -
                        datetime.fromisoformat(self.history['timestamps'][-2])).total_seconds()
            if time_diff > 0:
                disk_read_rate = ((self.history['disk_io'][-1]['read_bytes'] -
                                 self.history['disk_io'][-2]['read_bytes']) /
                                time_diff / (1024**2))
                disk_write_rate = ((self.history['disk_io'][-1]['write_bytes'] -
                                  self.history['disk_io'][-2]['write_bytes']) /
                                 time_diff / (1024**2))
                report.extend([
                    "",
                    "I/O Rates:",
                    f"Disk Read Rate: {disk_read_rate:.1f} MB/s",
                    f"Disk Write Rate: {disk_write_rate:.1f} MB/s"
                ])

        # Save history
        with open('resource_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return "\n".join(report)

def main():
    monitor = ResourceMonitor()
    print("System Resource Monitor")
    print("Monitoring system resources during data preparation...\n")

    while True:
        print("\033[2J\033[H")  # Clear screen
        print(monitor.monitor_resources())
        time.sleep(10)

if __name__ == "__main__":
    main()
