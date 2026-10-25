import socket
import json
import gzip
import struct
import time

class env:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8888,
                 connect_timeout=None, socket_timeout=None):
        self.IP = ip
        self.PORT = port
        self.sock = None
        self.closed = False  # 新增一个状态标志
        deadline = None if connect_timeout is None else time.monotonic()+connect_timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                remaining = None if deadline is None else max(0.01, deadline-time.monotonic())
                self.sock.settimeout(min(1.0, remaining) if remaining is not None else socket_timeout)
                self.sock.connect((self.IP, self.PORT))
                self.sock.settimeout(socket_timeout)
                break
            except OSError as error:
                self.sock.close()
                self.sock = None
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not connect to {ip}:{port}") from error
                time.sleep(0.1)

    def _send_data(self, data):
        if self.closed or not self.sock:
            raise ConnectionError("Socket is closed")
        try:
            json_str = json.dumps(data)
            compressed = gzip.compress(json_str.encode('utf-8'))
            header = struct.pack('!I', len(compressed))
            self.sock.sendall(header)
            self.sock.sendall(compressed)
        except Exception as e:
            self.close()
            raise ConnectionError("Platform send failed") from e

    def _receive_data(self):
        if self.closed or not self.sock:
            raise ConnectionError("Socket is closed")
        try:
            header = self._recv_all(4)
            if not header:
                return None
            data_len = struct.unpack('!I', header)[0]
            compressed_data = self._recv_all(data_len)
            if not compressed_data:
                return None
            json_str = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            raise ConnectionError("Platform receive failed") from e

    def _recv_all(self, n: int):
        data = bytearray()
        while len(data) < n:
            try:
                remaining = n - len(data)
                packet = self.sock.recv(remaining)
                if not packet:
                    raise ConnectionError("Platform closed the connection mid-message")
                data.extend(packet)
            except socket.timeout:
                raise TimeoutError("Platform response timed out")
        return bytes(data)

    def reset(self, reset_attribute: dict):
        self._send_data(reset_attribute)
        return self._receive_data()

    def step(self, action_attribute: dict):
        self._send_data(action_attribute)
        return self._receive_data()

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()
        self.sock = None
        self.closed = True
