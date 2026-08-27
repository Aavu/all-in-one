import sys
import subprocess
import torch

from pathlib import Path
from typing import List, Union

STEMS = ['bass', 'drums', 'other', 'vocals']
DEMIX_FORMATS = ('wav', 'mp3')


def check_demix_format(demix_format: str) -> str:
  """Validates a demixed-stem audio format, returning it unchanged."""
  if demix_format not in DEMIX_FORMATS:
    raise ValueError(f'demix_format must be one of {DEMIX_FORMATS}, got {demix_format!r}.')
  return demix_format


def demix(
  paths: List[Path],
  demix_dir: Path,
  device: Union[str, torch.device],
  demix_format: str = 'wav',
):
  """Demixes the audio file into its sources."""
  check_demix_format(demix_format)

  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if all((out_dir / f'{stem}.{demix_format}').is_file() for stem in STEMS):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    subprocess.run(
      [
        sys.executable, '-m', 'demucs.separate',
        '--out', demix_dir.as_posix(),
        '--name', 'htdemucs',
        '--device', str(device),
        # demucs writes wav unless told otherwise.
        *(['--mp3'] if demix_format == 'mp3' else []),
        *[path.as_posix() for path in todos],
      ],
      check=True,
    )

  return demix_paths
