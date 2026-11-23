"""TCP JPEG camera client — same read()/close() contract as RSCapture."""

import time


class TcpImageCapture:
    def __init__(self, name: str, bridge, image_key: str | None = None):
        self.name = name
        self.bridge = bridge
        self.image_key = image_key or name

    def read(self):
        deadline = time.time() + 5.0
        while time.time() < deadline:
            bgr = self.bridge.fetch_bgr(self.image_key)
            if bgr is not None:
                return True, bgr
            time.sleep(0.01)
        return False, None

    def close(self) -> None:
        pass
