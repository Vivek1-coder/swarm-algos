import threading
import time

class Topic:
    """Simple in-process pub/sub to simulate ROS topics."""
    def __init__(self):
        self._subs = []
        self._lock = threading.Lock()

    def publish(self, msg):
        with self._lock:
            for cb in list(self._subs):
                try:
                    cb(msg)
                except Exception:
                    # individual subscriber error shouldn't stop others
                    pass

    def subscribe(self, callback):
        with self._lock:
            self._subs.append(callback)

    def unsubscribe(self, callback):
        with self._lock:
            if callback in self._subs:
                self._subs.remove(callback)
