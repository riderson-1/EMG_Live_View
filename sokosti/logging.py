"""
logging.py - RunLogger: unified terminal + file logging for the Sokosti apps.

Both live apps (USB and BLE) print status/telemetry to the terminal. This
module centralizes that so the same lines are also written to a timestamped
log file, prefixed with a header block that records the full run metadata
and configuration (all CLI arguments) for reproducibility.

All writes are serialized under a lock so the BLE transport thread and the
matplotlib GUI thread can log concurrently without interleaving lines.
"""

import os
import threading
import time
from datetime import datetime


class RunLogger:
    """Writes timestamped lines to the terminal and to a log file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._file = open(path, "w", encoding="utf-8")

    def log(self, message):
        """Print a timestamped line to the terminal and append it to the log."""
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with self._lock:
            if self._closed:
                return
            print(line)
            self._file.write(line + "\n")
            self._file.flush()

    def write_header(self, title, metadata=None):
        """Write a run header block (title, start time, configuration)."""
        lines = ["=" * 60]
        lines.append(f" {title}")
        lines.append(f" Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        lines.append("=" * 60)
        if metadata:
            lines.append(" Configuration:")
            key_width = max(len(str(k)) for k in metadata)
            for key, value in metadata.items():
                lines.append(f"   {str(key).ljust(key_width)} = {value}")
            lines.append("=" * 60)
        lines.append("")
        block = "\n".join(lines)
        with self._lock:
            if self._closed:
                return
            print(block)
            self._file.write(block + "\n")
            self._file.flush()

    def close(self):
        """Close the log file. Safe to call more than once."""
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True