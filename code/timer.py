import time


class Timer:
    """Print a description of what's happening and the time that it took."""

    def __init__(self, description=None):
        self.description = description

    def __enter__(self):
        self.t1 = time.perf_counter()
        if self.description is not None:
            print(self.description)

    def __exit__(self, *args):
        self.t2 = time.perf_counter()
        print(f"{self.description}  Done in {self.t2 - self.t1:.2f} sec")
