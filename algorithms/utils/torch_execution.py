"""Apply CPU thread counts at phase boundaries, with exception-safe restoration."""

from contextlib import contextmanager
import torch


@contextmanager
def torch_thread_count(count):
    # The setting affects the process. Only enter at a barrier, with no model
    # operations running in workers; socket workers do not change this setting.
    previous = torch.get_num_threads()
    changed = previous != count
    if changed:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        if changed:
            torch.set_num_threads(previous)
