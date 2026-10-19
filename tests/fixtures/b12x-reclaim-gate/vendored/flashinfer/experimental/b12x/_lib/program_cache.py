"""Minimal stand-in for b12x._lib.program_cache."""

CALLS = []


def evict_unretained(keep):
    CALLS.append(frozenset(keep))
    return 0


def reclaim_device_memory(keep, stack_limit=None):
    CALLS.append(("reclaim_device_memory", frozenset(keep)))
    return None
