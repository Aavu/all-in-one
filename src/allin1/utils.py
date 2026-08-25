import re

from multiprocessing import Pool
from pathlib import Path
from .typings import PathLike, AnalysisResult


def compact_json_number_array(json_str: str):
  """Compact numbers (including floats) in JSON arrays to be on the same line."""
  return re.sub(
    r'(\[\n(?:\s*\d+(\.\d+)?,\n)+\s*\d+(\.\d+)?\n\s*\])',
    lambda m: m.group(1).replace('\n', '').replace(' ', ''),
    json_str
  )


def mkpath(path: PathLike):
  return Path(path).expanduser().resolve()


def load_result(
  path: PathLike,
  load_activations: bool = True,
  load_embeddings: bool = True,
) -> AnalysisResult:
  path = mkpath(path)
  result = AnalysisResult.from_json(
    path,
    load_activations=load_activations,
    load_embeddings=load_embeddings,
  )
  return result


def imap_maybe_parallel(fn, items, multiprocess: bool, unordered: bool = False):
  """Map `fn` over `items`, in a process pool only when that actually helps.

  Returns `(iterator, pool)`; close the pool with `close_pool(pool)` once the
  iterator is drained.

  A pool is skipped for a single item. That is a small speedup -- spinning up
  one worker per CPU to run one task is pure overhead -- but the real reason is
  robustness. On macOS and Windows `multiprocessing` defaults to the `spawn`
  start method, and every spawned child re-imports the caller's `__main__`
  module. A script like

      import allin1
      allin1.analyze('song.mp3')      # module level, no __main__ guard

  therefore re-runs `analyze()` inside each child, which tries to spawn again
  and dies with "An attempt has been made to start a new process before the
  current process has finished its bootstrapping phase", leaving the parent
  hanging on a pool whose workers are all dead.

  Analysing one file is the common case and now never hits that. Batch callers
  still get the pool, and for them the standard Python rule applies -- guard the
  entry point:

      if __name__ == '__main__':
          allin1.analyze(['a.mp3', 'b.mp3'])

  or pass `multiprocess=False`.
  """
  items = list(items)
  if not multiprocess or len(items) < 2:
    return map(fn, items), None
  pool = Pool()
  iterator = pool.imap_unordered(fn, items) if unordered else pool.imap(fn, items)
  return iterator, pool


def close_pool(pool) -> None:
  """Close and join a pool from `imap_maybe_parallel`, if there is one."""
  if pool is not None:
    pool.close()
    pool.join()
