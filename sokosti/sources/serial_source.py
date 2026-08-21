"""
serial_source.py - USB CDC-ACM byte transport (SerialSource).

A background thread reads bounded chunks from a pyserial port and feeds
them into a FrameParser. Complete packets are dispatched into a
PacketSink. This replaces the SerialReader class from emg_live_plot.py.
"""

import sys
import threading

import serial


class SerialSource(threading.Thread):
    """Reads bytes from a serial port and pushes packets into a sink."""

    def __init__(self, ser, parser, sink, read_size=512):
        super().__init__(daemon=True)
        self.ser = ser
        self.parser = parser
        self.sink = sink
        self.read_size = read_size
        self.stop_flag = threading.Event()

    def run(self):
        print("Serial reader started...", file=sys.stderr)
        while not self.stop_flag.is_set():
            try:
                # Read bounded chunks so a large host-side USB backlog does not
                # monopolize the GUI thread's companion reader thread.
                chunk = self.ser.read(min(self.ser.in_waiting or 128, self.read_size))
            except serial.SerialException as e:
                print(f"Serial error: {e}", file=sys.stderr)
                break

            if not chunk:
                continue

            for kind, packet in self.parser.feed(chunk):
                if kind == "emg":
                    self.sink.add_emg(packet)
                else:
                    self.sink.add_imu(packet)

    def stop(self):
        self.stop_flag.set()