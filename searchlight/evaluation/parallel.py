"""Ordered, bounded spawn pools for CPU evaluation work."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing as mp
import os

from tqdm.auto import tqdm

_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def worker_count(workers, tasks=None):
    """Auto uses up to 32 available CPUs; never start more processes than jobs."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    if workers == 'auto':
        count = min(32, available)
    elif isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1:
        count = min(workers, available)
    else:
        raise ValueError('workers must be a positive integer or "auto"')
    return min(count, max(1, tasks)) if tasks is not None else count


def initialize_worker():
    """Give each worker one CPU, including JAX's compilation/runtime pools."""
    for name in _THREAD_ENV:
        os.environ[name] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.setdefault('JAX_ENABLE_X64', 'true')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        identity = mp.current_process()._identity
        index = (identity[-1] - 1) if identity else 0
        os.sched_setaffinity(0, {cpus[index % len(cpus)]})


@contextmanager
def pool(workers, *, initializer=initialize_worker, initargs=()):
    # Spawn imports NumPy before its initializer. Set thread limits in the
    # inherited environment before starting children, then restore the parent.
    previous = {name: os.environ.get(name) for name in _THREAD_ENV}
    try:
        for name in _THREAD_ENV:
            os.environ[name] = '1'
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=initializer, initargs=initargs) as executor:
            yield executor
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def process_map(function, jobs, *, workers=1, desc=None, progress=False,
                initializer=initialize_worker, initargs=()):
    """Display completed-job progress while preserving input order and errors."""
    jobs = list(jobs)
    count = worker_count(workers, len(jobs))
    if count == 1:
        return [function(job) for job in tqdm(jobs, desc=desc, disable=not progress)]
    results = [None] * len(jobs)
    with pool(count, initializer=initializer, initargs=initargs) as executor:
        pending = {executor.submit(function, job): i for i, job in enumerate(jobs)}
        try:
            for future in tqdm(as_completed(pending), total=len(jobs), desc=f'{desc} ({count} workers)', disable=not progress):
                results[pending[future]] = future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results
