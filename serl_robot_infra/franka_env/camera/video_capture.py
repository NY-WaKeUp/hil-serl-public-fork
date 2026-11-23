import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self._last = None
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self._last = frame
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return self.q.get(timeout=5)

    def read_latest(self):
        """Return newest frame; block only until the first frame arrives."""
        if self._last is None:
            return self.read()
        try:
            while True:
                self._last = self.q.get_nowait()
        except queue.Empty:
            pass
        return self._last

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
