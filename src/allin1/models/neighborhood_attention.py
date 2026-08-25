"""Neighborhood attention ops for the DINAT encoder, with no hard natten dependency.

Background
----------
allin1's DINAT encoder uses natten's *unfused* neighborhood attention: a QK pass
that adds a learned relative positional bias (RPB) to the neighborhood scores, a
softmax, then an AV pass.

    scores = na1d_qk(q, k, kernel_size, dilation, rpb=rpb)
    probs  = softmax(scores)
    out    = na1d_av(probs, v, kernel_size, dilation)

Those ops were named `natten1dqkrpb` / `natten1dav` up to natten 0.15, renamed to
`na1d_qk` / `na1d_av` in 0.16, and **removed outright in natten 0.20**. From 0.20
on, natten exposes only the fused `na1d`/`na2d`/`na3d`, and none of its backends
(`cutlass-fna`, `hopper-fna`, `blackwell-fna`, `flex-fna`) accept an additive
bias -- there is no `rpb`, no `attn_bias`, no `score_mod` anywhere in
`natten/backends/`. Since `rpb` is a learned parameter in all 22 of allin1's
attention modules and comes out of the pretrained checkpoint, the fused ops
cannot express this model. Re-pointing imports at the new names does not help;
the capability is gone, not renamed.

That left allin1 pinned to natten 0.17.5, which in turn pins torch <= 2.6
(0.17.5 imports `torch.cuda._device_t`, removed after 2.6) and needs a ~10 minute
source build with a patched `setup.py` to reach sm_120.

So this module reimplements the four ops in plain PyTorch. That is a cheap trade
at allin1's sizes: the 1D time attention is
`[batch * 4 stems, heads=2, frames, head_dim=12]` with `kernel_size=5`, so the
score tensor is 5 floats per query, and the 2D instrument attention over
`(4 stems, frames)` is 25. The memory blow-up that fused neighborhood-attention
kernels exist to avoid never materialises here, so gather + `matmul` is enough --
and it runs on whatever torch runs on: any version, CPU, CUDA (incl. sm_120),
or MPS, with no compiler and no extension.

natten is still used when it is installed *and* still has the unfused ops
(0.16-0.17.x), so existing environments keep their native CUDA kernels and their
exact previous numerics. Override with `ALLIN1_NA_BACKEND=torch|natten|auto`.

Fidelity
--------
`_window_start` and `_pb_start` are vectorised transcriptions of natten 0.17.5's
`csrc/include/natten/cpu/naive/natten_cpu_commons.h`, and the neighbor / bias /
output index arithmetic follows `pointwise_neighborhood_{1,2}d.hpp` and
`neighborhood_neighborhood_{1,2}d.hpp` (kernel index is row-major,
`ki * kernel_size_1 + kj`). tests/test_neighborhood_attention.py checks the torch
path against a real natten build whenever one is importable.

Two deliberate deviations, both in cases allin1 never reaches:

* `is_causal` and `additional_keys`/`additional_values` raise NotImplementedError
  rather than being silently ignored.
* When `kernel_size * dilation > length` the window runs off the end. natten's
  bias kernel reads out of bounds there (undefined behaviour); this clamps the
  gather and masks with -inf. `_DinatLayerNd.maybe_pad` pads the input up to
  `kernel_size * dilation` precisely so this cannot happen.
"""

import math
import os
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
  'na1d_qk', 'na1d_av', 'na2d_qk', 'na2d_av',
  'na1d_qk_torch', 'na1d_av_torch', 'na2d_qk_torch', 'na2d_av_torch',
  'BACKEND', 'set_gather_budget',
]

IntOrSeq = Union[int, Sequence[int]]

# Cap on the element count of the gathered key/value temporaries. The 2D ops
# chunk their query axis to stay under this, which bounds peak memory
# independently of clip length. 8M elements is 32 MB in float32.
_GATHER_BUDGET = 1 << 23


def set_gather_budget(elements: int) -> None:
  """Override the per-temporary element budget used to size gather chunks."""
  global _GATHER_BUDGET
  if elements < 1:
    raise ValueError(f'budget must be positive, got {elements}')
  _GATHER_BUDGET = int(elements)


def _as_tuple(value: IntOrSeq, n: int, name: str) -> Tuple[int, ...]:
  if isinstance(value, int):
    return (value,) * n
  out = tuple(int(v) for v in value)
  if len(out) != n:
    raise ValueError(f'{name} must be an int or a length-{n} sequence, got {value!r}')
  return out


def _reject_unsupported(is_causal, additional, additional_name: str) -> None:
  if additional is not None:
    raise NotImplementedError(
      f'{additional_name} (neighborhood cross-attention) is not implemented in the '
      'torch backend; allin1 does not use it.'
    )
  if is_causal is None or is_causal is False:
    return
  flags = (is_causal,) if isinstance(is_causal, bool) else tuple(is_causal)
  if any(flags):
    raise NotImplementedError(
      'is_causal=True is not implemented in the torch backend; allin1 does not use it.'
    )


def _window_start(length: int, kernel_size: int, dilation: int, device) -> Tensor:
  """natten's `get_window_start` for every query index, is_causal=False.

  Returns the first neighbor index for each of `length` queries, shape [length].
  """
  neighborhood = kernel_size // 2
  index = torch.arange(length, device=device)

  # Dilated NA runs independently within each residue class mod `dilation`;
  # natten calls these per-dilation positions.
  dilation_idx = index % dilation
  index_pdp = index // dilation
  length_pdp = (length + dilation - 1) // dilation
  num_padded = length_pdp * dilation - length
  # Residue classes past the end of the padded grid are one element shorter.
  length_pdp = length_pdp - (dilation_idx >= dilation - num_padded).long()

  start_pdp = torch.clamp(index_pdp - neighborhood, min=0) + (
    index_pdp + neighborhood >= length_pdp
  ).long() * (length_pdp - index_pdp - neighborhood - 1)

  return start_pdp * dilation + dilation_idx


def _pb_start(length: int, kernel_size: int, dilation: int, device) -> Tensor:
  """natten's `get_pb_start` for every query index, shape [length].

  Offset into the `2 * kernel_size - 1` bias table for each query's first
  neighbor; neighbor `ki` reads `rpb[pb_start + ki]`.
  """
  neighborhood = kernel_size // 2
  index = torch.arange(length, device=device)

  if dilation <= 1:
    return (
      neighborhood
      + (index < neighborhood).long() * (neighborhood - index)
      + (index + neighborhood >= length).long() * (length - index - 1 - neighborhood)
    )

  # natten checks the left edge first and returns, so `left` wins over `right`
  # when a residue class is short enough to satisfy both.
  left = (index - neighborhood * dilation) < 0
  right = (index + neighborhood * dilation) >= length
  out = torch.full_like(index, neighborhood)
  out = torch.where(left, kernel_size - 1 - index // dilation, out)
  return torch.where(right & ~left, (length - index - 1) // dilation, out)


def _neighbors(length: int, kernel_size: int, dilation: int, device) -> Tuple[Tensor, Tensor]:
  """Neighbor indices [length, kernel_size] and a validity mask of the same shape.

  Invalid entries appear only when `kernel_size * dilation > length`. The indices
  are clamped, so they are always safe to gather with.
  """
  offsets = torch.arange(kernel_size, device=device) * dilation
  neighbors = _window_start(length, kernel_size, dilation, device)[:, None] + offsets[None, :]
  valid = neighbors < length
  return neighbors.clamp_(min=0, max=length - 1), valid


def _bias_index(length: int, kernel_size: int, dilation: int, device) -> Tensor:
  """Bias-table indices [length, kernel_size] for each query's neighbors."""
  offsets = torch.arange(kernel_size, device=device)
  return _pb_start(length, kernel_size, dilation, device)[:, None] + offsets[None, :]


def _chunk_bounds(total: int, cost_per_index: int):
  """Split range(total) so each chunk's temporary stays under the gather budget."""
  step = max(1, min(total, _GATHER_BUDGET // max(1, cost_per_index)))
  for begin in range(0, total, step):
    yield begin, min(begin + step, total)


def na1d_qk_torch(
  query: Tensor,
  key: Tensor,
  kernel_size: IntOrSeq,
  dilation: IntOrSeq = 1,
  additional_keys: Optional[Tensor] = None,
  is_causal=False,
  rpb: Optional[Tensor] = None,
) -> Tensor:
  """1-D neighborhood QK with optional RPB.

  [batch, heads, length, dim] x2 -> [batch, heads, length, kernel_size].
  """
  _reject_unsupported(is_causal, additional_keys, 'additional_keys')
  if query.dim() != 4:
    raise ValueError(f'na1d_qk expects [batch, heads, length, dim], got {tuple(query.shape)}')

  kernel, = _as_tuple(kernel_size, 1, 'kernel_size')
  dil, = _as_tuple(dilation, 1, 'dilation')
  batch, heads, length, dim = query.shape
  device = query.device

  neighbors, valid = _neighbors(length, kernel, dil, device)
  keys = key.index_select(2, neighbors.reshape(-1)).view(batch, heads, length, kernel, dim)
  scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)

  if rpb is not None:
    scores = scores + rpb[:, _bias_index(length, kernel, dil, device)].unsqueeze(0)
  if not bool(valid.all()):
    scores = scores.masked_fill(~valid, -math.inf)
  return scores


def na1d_av_torch(
  attn: Tensor,
  value: Tensor,
  kernel_size: IntOrSeq,
  dilation: IntOrSeq = 1,
  additional_values: Optional[Tensor] = None,
  is_causal=False,
) -> Tensor:
  """1-D neighborhood AV.

  [batch, heads, length, kernel_size] x [batch, heads, length, dim]
  -> [batch, heads, length, dim].
  """
  _reject_unsupported(is_causal, additional_values, 'additional_values')
  if attn.dim() != 4:
    raise ValueError(f'na1d_av expects [batch, heads, length, kernel], got {tuple(attn.shape)}')

  kernel, = _as_tuple(kernel_size, 1, 'kernel_size')
  dil, = _as_tuple(dilation, 1, 'dilation')
  batch, heads, length, dim = value.shape
  device = value.device

  neighbors, valid = _neighbors(length, kernel, dil, device)
  if not bool(valid.all()):
    # natten's AV loop stops at the window end, i.e. drops the invalid tail.
    attn = attn * valid
  values = value.index_select(2, neighbors.reshape(-1)).view(batch, heads, length, kernel, dim)
  return torch.matmul(attn.unsqueeze(-2), values).squeeze(-2)


def na2d_qk_torch(
  query: Tensor,
  key: Tensor,
  kernel_size: IntOrSeq,
  dilation: IntOrSeq = 1,
  additional_keys: Optional[Tensor] = None,
  is_causal=False,
  rpb: Optional[Tensor] = None,
) -> Tensor:
  """2-D neighborhood QK with optional RPB.

  [batch, heads, X, Y, dim] x2 -> [batch, heads, X, Y, kernel_x * kernel_y],
  kernel index row-major as `ki * kernel_y + kj`, matching natten's layout.
  """
  _reject_unsupported(is_causal, additional_keys, 'additional_keys')
  if query.dim() != 5:
    raise ValueError(f'na2d_qk expects [batch, heads, X, Y, dim], got {tuple(query.shape)}')

  kernel_x, kernel_y = _as_tuple(kernel_size, 2, 'kernel_size')
  dil_x, dil_y = _as_tuple(dilation, 2, 'dilation')
  batch, heads, size_x, size_y, dim = query.shape
  device = query.device

  nbr_x, valid_x = _neighbors(size_x, kernel_x, dil_x, device)
  nbr_y, valid_y = _neighbors(size_y, kernel_y, dil_y, device)
  bias_x = _bias_index(size_x, kernel_x, dil_x, device) if rpb is not None else None
  bias_y = _bias_index(size_y, kernel_y, dil_y, device) if rpb is not None else None

  scores = query.new_empty(batch, heads, size_x, size_y, kernel_x * kernel_y)

  # Gather the X axis one kernel row at a time and chunk the Y axis, so the
  # largest temporary is [batch, heads, X, y_chunk, kernel_y, dim].
  for begin, end in _chunk_bounds(size_y, batch * heads * size_x * kernel_y * dim):
    nbr_y_chunk = nbr_y[begin:end]
    query_chunk = query[:, :, :, begin:end, :].unsqueeze(-1)
    for ki in range(kernel_x):
      keys = key.index_select(2, nbr_x[:, ki])
      keys = keys.index_select(3, nbr_y_chunk.reshape(-1))
      keys = keys.view(batch, heads, size_x, end - begin, kernel_y, dim)
      block = torch.matmul(keys, query_chunk).squeeze(-1)
      if rpb is not None:
        block = block + rpb[:, bias_x[:, ki], :][:, :, bias_y[begin:end]].unsqueeze(0)
      scores[:, :, :, begin:end, ki * kernel_y:(ki + 1) * kernel_y] = block

  valid = (valid_x[:, None, :, None] & valid_y[None, :, None, :]).reshape(
    size_x, size_y, kernel_x * kernel_y
  )
  if not bool(valid.all()):
    scores = scores.masked_fill(~valid, -math.inf)
  return scores


def na2d_av_torch(
  attn: Tensor,
  value: Tensor,
  kernel_size: IntOrSeq,
  dilation: IntOrSeq = 1,
  additional_values: Optional[Tensor] = None,
  is_causal=False,
) -> Tensor:
  """2-D neighborhood AV.

  [batch, heads, X, Y, kernel_x * kernel_y] x [batch, heads, X, Y, dim]
  -> [batch, heads, X, Y, dim].
  """
  _reject_unsupported(is_causal, additional_values, 'additional_values')
  if attn.dim() != 5:
    raise ValueError(f'na2d_av expects [batch, heads, X, Y, kernel], got {tuple(attn.shape)}')

  kernel_x, kernel_y = _as_tuple(kernel_size, 2, 'kernel_size')
  dil_x, dil_y = _as_tuple(dilation, 2, 'dilation')
  batch, heads, size_x, size_y, dim = value.shape
  device = value.device

  nbr_x, valid_x = _neighbors(size_x, kernel_x, dil_x, device)
  nbr_y, valid_y = _neighbors(size_y, kernel_y, dil_y, device)

  valid = (valid_x[:, None, :, None] & valid_y[None, :, None, :]).reshape(
    size_x, size_y, kernel_x * kernel_y
  )
  if not bool(valid.all()):
    attn = attn * valid

  out = torch.zeros_like(value)
  for begin, end in _chunk_bounds(size_y, batch * heads * size_x * kernel_y * dim):
    nbr_y_chunk = nbr_y[begin:end]
    acc = None
    for ki in range(kernel_x):
      values = value.index_select(2, nbr_x[:, ki])
      values = values.index_select(3, nbr_y_chunk.reshape(-1))
      values = values.view(batch, heads, size_x, end - begin, kernel_y, dim)
      weights = attn[:, :, :, begin:end, ki * kernel_y:(ki + 1) * kernel_y]
      block = torch.matmul(weights.unsqueeze(-2), values).squeeze(-2)
      acc = block if acc is None else acc + block
    out[:, :, :, begin:end, :] = acc
  return out


def _resolve_backend():
  """Pick the op implementations: natten's own unfused kernels, or torch.

  `ALLIN1_NA_BACKEND` overrides the choice:
    auto (default) -- natten if it still has the unfused ops, else torch
    natten         -- require natten's unfused ops; raise if unavailable
    torch          -- always use the torch implementations
  """
  requested = os.environ.get('ALLIN1_NA_BACKEND', 'auto').strip().lower()
  if requested not in ('auto', 'natten', 'torch'):
    raise ValueError(
      f'ALLIN1_NA_BACKEND must be one of auto/natten/torch, got {requested!r}'
    )

  torch_ops = (na1d_qk_torch, na1d_av_torch, na2d_qk_torch, na2d_av_torch)
  if requested == 'torch':
    return 'torch', torch_ops

  natten_ops = None
  try:
    # natten 0.17.5 and earlier probe CUDA at import time, so a broken or
    # CPU-only install can raise something other than ImportError here.
    from natten.functional import na1d_av, na1d_qk, na2d_av, na2d_qk
    natten_ops = (na1d_qk, na1d_av, na2d_qk, na2d_av)
  except Exception:
    natten_ops = None

  if natten_ops is not None:
    return 'natten', natten_ops
  if requested == 'natten':
    raise ImportError(
      'ALLIN1_NA_BACKEND=natten was requested, but natten does not expose the '
      'unfused ops allin1 needs (na1d_qk/na1d_av/na2d_qk/na2d_av). natten 0.20 '
      'removed them; install natten 0.17.x or unset ALLIN1_NA_BACKEND.'
    )
  return 'torch', torch_ops


BACKEND, (na1d_qk, na1d_av, na2d_qk, na2d_av) = _resolve_backend()
