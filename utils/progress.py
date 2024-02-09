from tqdm import tqdm as _tqdm
import os
from typing import Optional
from utils.time import get_current_datetime_str
import time

NO_TQDM = bool(os.environ.get("NO_TQDM", ""))
PRINT_INTERVAL = int(os.environ.get("PRINT_INTERVAL", "1"))


def tqdm(iter, desc: str):
    if not NO_TQDM:
        return _tqdm(iter, desc=desc)

    class CLITqdm:
        def __init__(self, iter, desc: str) -> None:
            self._iter = iter
            self._desc = desc
            self._print_interval = PRINT_INTERVAL
            self._i = 0

        def __iter__(self):
            for ele in self._iter:
                if self._i % self._print_interval == 0:
                    print(
                        f"----CLI-tqdm: {self._i}th {self._desc} started at {get_current_datetime_str(no_micro=True)}",
                        flush=True,
                    )
                yield ele
                self._i += 1

    return CLITqdm(iter, desc=desc)


class TqdmBar:
    def __init__(self, total: int, desc: str):
        if not NO_TQDM:
            self._tqdm = _tqdm(total=total, desc=desc)
        else:
            self._tqdm = None
        self._desc = desc
        self._print_interval = PRINT_INTERVAL
        self._i = 0

    def __enter__(self):
        if self._tqdm:
            self._tqdm.__enter__()
            return self._tqdm
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self._tqdm:
            self._tqdm.__exit__(exception_type, exception_value, exception_traceback)
            return

    def update(self, *args, **kwargs):
        if self._tqdm:
            self._tqdm.update(*args, **kwargs)
            return

        if self._i % self._print_interval == 0:
            print(
                f"----CLI-tqdm bar: {self._i}th {self._desc} end at {get_current_datetime_str(no_micro=True)}",
                flush=True,
            )
        self._i += 1


if __name__ == "__main__":
    n = 0
    for i in tqdm(range(int(1e7)), desc="training..."):
        n += i * i
    print(n)

    n = 0
    with TqdmBar(total=int(1e7), desc="training in bar...") as pbar:
        for i in range(int(1e7)):
            n += i * i
            pbar.update(1)
    print(n)
