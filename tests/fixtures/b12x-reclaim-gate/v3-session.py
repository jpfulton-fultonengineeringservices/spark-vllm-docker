"""Fixture mirroring flashinfer-vendored b12x preparation/session.py.

Only the two reclamation sites patched by mods/b12x-reclaim-gate matter here;
everything else is a minimal stand-in so the file compiles and the exact
anchors used by the mod are present verbatim.
"""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager

from b12x._lib.program_cache import evict_unretained

logger = logging.getLogger("b12x.preparation")


class PreparationTiming:
    def __init__(self, phase, rank=None):
        self.phase = phase
        self.rank = rank

    @contextmanager
    def span(self, name):
        yield

    def record(self, name, **fields):
        pass


class PreparedPrograms:
    def __init__(self, name):
        self.programs = (name,)


class PreparedPlan:
    def __init__(self, name):
        self.prepared = PreparedPrograms(name)


class PreparationSession:
    def __init__(self, plans=()):
        self._plans = list(plans)
        self._tuning_rank = 0

    def _synchronize(self):
        return None

    def _reclaim_programs(self):
        import os

        if os.environ.get("B12X_SKIP_RECLAIM") == "1":
            # b12x-reclaim-gate v3: skip the entire reclamation sweep.
            logger.warning_once(
                "b12x-reclaim-gate: B12X_SKIP_RECLAIM=1; skipped the whole "
                "reclamation sweep (keep-walk + evict_unretained + gc.collect)"
            )
            return
        timing = PreparationTiming("cleanup", rank=self._tuning_rank)
        keep = frozenset(program for plan in self._plans if plan.prepared is not None
                         for program in plan.prepared.programs)
        with timing.span("program_reclamation"):
            removed = evict_unretained(keep)
            gc.collect()
        timing.record("complete", evicted_entries=removed)
    @contextmanager
    def capture(self):
        yield


class PreparationResult:
    def __init__(self, keep):
        self.keep = keep


class PreparationJob:
    def __init__(self, session, program_cache):
        self.session = session
        self._program_cache = program_cache
        self._benchmarked = True
        self._stack_limit = None

    def _run(self):
        result = PreparationResult(None)
        from b12x._lib.program_cache import evict_unretained

        keep = frozenset(
            program
            for plan in self.session._plans
            if plan.prepared is not None
            for program in plan.prepared.programs
        )
        self.session._synchronize()
        self._program_cache.clear()
        evict_unretained(keep)
        result = PreparationResult(keep)
        if self._benchmarked:
            from b12x._lib.program_cache import reclaim_device_memory

            self.session._synchronize()
            reclaim_device_memory(keep, stack_limit=self._stack_limit)
        return result
