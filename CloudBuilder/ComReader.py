from typing import Any
import serial
import json
from threading import Thread, Lock
from copy import deepcopy


class ComReader:
    def __init__(self,
                 port: str,
                 baud_rate: int = 115200,
                 timeout: int = 2,
                 codec: str = "utf-8"
                 ) -> None:
        self.baud_rate: int = baud_rate
        self.port = port
        self.timeout = timeout
        self.codec = codec
        self.ser: serial.Serial = None
        self.__data__: Any = None
        self.data_read_lock = Lock()
        self.read_thread: Thread = None
        self.READ_CONTINUE: bool = False

    def __del__(self):
        self.READ_CONTINUE = False
        if not self.ser.closed:
            self.ser.close()
        if (self.read_thread.is_alive()):
            self.read_thread.join()

    @property
    def data(self):
        with self.data_read_lock:
            return deepcopy(self.__data__)

    def start_read(self):
        self.ser = serial.Serial(
            self.port, self.baud_rate, timeout=self.timeout)
        self.READ_CONTINUE = True
        self.read_thread = Thread(target=self.__read__)
        self.read_thread.start()

    def stop_read(self):
        self.READ_CONTINUE = False
        self.read_thread.join()
        self.ser.close()

    def __read__(self):
        while self.READ_CONTINUE:
            line = self.ser.readline().decode(self.codec).strip()
            if line:
                try:
                    self.__data__ = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Invalid json format:\n{line}")
