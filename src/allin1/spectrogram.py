import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from .utils import close_pool, imap_maybe_parallel


def extract_spectrograms(
  demix_paths: List[Path],
  spec_dir: Path,
  multiprocess: bool = True,
  demix_format: str = 'wav',
):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    # Define a pre-processing chain, which is copied from madmom.
    frames = FramedSignalProcessor(
      frame_size=2048,
      fps=int(44100 / 441)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
      num_bands=12,
      fmin=30,
      fmax=17000,
      norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([frames, stft, filt, spec])

    # Process all tracks, in a pool only when there is more than one.
    iterator, pool = imap_maybe_parallel(
      _extract_spectrogram,
      [(src, dst, processor, demix_format) for src, dst in todos],
      multiprocess,
    )
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass
    close_pool(pool)

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor, str]):
  src, dst, processor, demix_format = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  # str(), not Path: madmom reads wav itself but shells out to ffmpeg for
  # everything else, and its ffmpeg path rejects anything that is not a string.
  sig_bass = Signal(str(src / f'bass.{demix_format}'), num_channels=1)
  sig_drums = Signal(str(src / f'drums.{demix_format}'), num_channels=1)
  sig_other = Signal(str(src / f'other.{demix_format}'), num_channels=1)
  sig_vocals = Signal(str(src / f'vocals.{demix_format}'), num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)
