"""Tests for the torch neighborhood-attention fallback.

Two independent checks:

1. Against a naive nested-loop reference in this file, which is a line-by-line
   transcription of natten 0.17.5's C++ kernels
   (`csrc/include/natten/cpu/naive/`). This needs no natten install, so it runs
   everywhere and is what actually pins the semantics -- boundary window
   arithmetic, the relative-position-bias table lookup, and the row-major
   `ki * kernel_y + kj` kernel index.

2. Against natten itself, whenever an installed natten still exposes the unfused
   ops (0.16-0.17.x). Skipped otherwise, which includes every natten >= 0.20.

Run:  pytest tests/test_neighborhood_attention.py
"""

import itertools

import pytest
import torch

from allin1.models import neighborhood_attention as na

# natten's own unfused ops, if this environment has them. `na.BACKEND` already
# reflects whether they were found, but import them directly so the comparison
# does not depend on which backend happened to win.
try:
  from natten.functional import na1d_av as natten_na1d_av
  from natten.functional import na1d_qk as natten_na1d_qk
  from natten.functional import na2d_av as natten_na2d_av
  from natten.functional import na2d_qk as natten_na2d_qk
  HAS_NATTEN_UNFUSED = True
except Exception:
  HAS_NATTEN_UNFUSED = False

requires_natten = pytest.mark.skipif(
  not HAS_NATTEN_UNFUSED,
  reason='needs a natten with the unfused ops (0.16-0.17.x); 0.20+ removed them',
)

# float64 keeps these about semantics rather than accumulation order.
EXACT = {'rtol': 1e-12, 'atol': 1e-12}


# ---------------------------------------------------------------------------
# Naive reference: natten 0.17.5's CPU kernels, transcribed literally.
# ---------------------------------------------------------------------------

def ref_window_start(index, length, kernel_size, dilation):
  """`get_window_start` from natten_cpu_commons.h, is_causal=False."""
  neighborhood = kernel_size // 2
  dilation_idx = index % dilation
  index_pdp = index // dilation
  length_pdp = (length + dilation - 1) // dilation
  num_padded = length_pdp * dilation - length
  if dilation_idx >= dilation - num_padded:
    length_pdp -= 1
  start = max(index_pdp - neighborhood, 0) + (
    (index_pdp + neighborhood >= length_pdp) * (length_pdp - index_pdp - neighborhood - 1)
  )
  return start * dilation + dilation_idx


def ref_pb_start(index, length, kernel_size, dilation):
  """`get_pb_start` from natten_cpu_commons.h."""
  neighborhood = kernel_size // 2
  if dilation <= 1:
    return (
      neighborhood
      + (index < neighborhood) * (neighborhood - index)
      + (index + neighborhood >= length) * (length - index - 1 - neighborhood)
    )
  if index - neighborhood * dilation < 0:
    return kernel_size - 1 - index // dilation
  if index + neighborhood * dilation >= length:
    return (length - index - 1) // dilation
  return neighborhood


def ref_na1d_qk(query, key, kernel_size, dilation, rpb=None):
  batch, heads, length, _ = query.shape
  out = torch.zeros(batch, heads, length, kernel_size, dtype=query.dtype)
  for b, h, i in itertools.product(range(batch), range(heads), range(length)):
    start = ref_window_start(i, length, kernel_size, dilation)
    end = min(length, start + kernel_size * dilation)
    pb = ref_pb_start(i, length, kernel_size, dilation)
    for ki in range(kernel_size):
      neighbor = start + ki * dilation
      if neighbor < end:
        value = torch.dot(query[b, h, i], key[b, h, neighbor])
        if rpb is not None:
          value = value + rpb[h, pb + ki]
      else:
        value = torch.tensor(float('-inf'), dtype=query.dtype)
      out[b, h, i, ki] = value
  return out


def ref_na1d_av(attn, value, kernel_size, dilation):
  batch, heads, length, dim = value.shape
  out = torch.zeros(batch, heads, length, dim, dtype=value.dtype)
  for b, h, i in itertools.product(range(batch), range(heads), range(length)):
    start = ref_window_start(i, length, kernel_size, dilation)
    end = min(length, start + kernel_size * dilation)
    for ki in range(kernel_size):
      neighbor = start + ki * dilation
      if neighbor < end:
        out[b, h, i] += attn[b, h, i, ki] * value[b, h, neighbor]
  return out


def ref_na2d_qk(query, key, kernel_size, dilation, rpb=None):
  batch, heads, size_x, size_y, _ = query.shape
  out = torch.zeros(batch, heads, size_x, size_y, kernel_size**2, dtype=query.dtype)
  for b, h, i, j in itertools.product(
    range(batch), range(heads), range(size_x), range(size_y)
  ):
    start_i = ref_window_start(i, size_x, kernel_size, dilation)
    start_j = ref_window_start(j, size_y, kernel_size, dilation)
    end_i = min(size_x, start_i + kernel_size * dilation)
    end_j = min(size_y, start_j + kernel_size * dilation)
    pb_i = ref_pb_start(i, size_x, kernel_size, dilation)
    pb_j = ref_pb_start(j, size_y, kernel_size, dilation)
    for ki, kj in itertools.product(range(kernel_size), range(kernel_size)):
      neighbor_i = start_i + ki * dilation
      neighbor_j = start_j + kj * dilation
      if neighbor_i < end_i and neighbor_j < end_j:
        value = torch.dot(query[b, h, i, j], key[b, h, neighbor_i, neighbor_j])
        if rpb is not None:
          value = value + rpb[h, pb_i + ki, pb_j + kj]
      else:
        value = torch.tensor(float('-inf'), dtype=query.dtype)
      # Row-major kernel index, exactly as natten writes it.
      out[b, h, i, j, ki * kernel_size + kj] = value
  return out


def ref_na2d_av(attn, value, kernel_size, dilation):
  batch, heads, size_x, size_y, dim = value.shape
  out = torch.zeros(batch, heads, size_x, size_y, dim, dtype=value.dtype)
  for b, h, i, j in itertools.product(
    range(batch), range(heads), range(size_x), range(size_y)
  ):
    start_i = ref_window_start(i, size_x, kernel_size, dilation)
    start_j = ref_window_start(j, size_y, kernel_size, dilation)
    end_i = min(size_x, start_i + kernel_size * dilation)
    end_j = min(size_y, start_j + kernel_size * dilation)
    for ki, kj in itertools.product(range(kernel_size), range(kernel_size)):
      neighbor_i = start_i + ki * dilation
      neighbor_j = start_j + kj * dilation
      if neighbor_i < end_i and neighbor_j < end_j:
        weight = attn[b, h, i, j, ki * kernel_size + kj]
        out[b, h, i, j] += weight * value[b, h, neighbor_i, neighbor_j]
  return out


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

def cases_1d():
  """Includes the exactly-tight window, where boundary handling is trickiest."""
  for kernel, dilation in itertools.product([3, 5, 7], [1, 2, 3, 4, 8]):
    window = kernel * dilation
    for length in sorted({window, window + 1, window + 7, window * 2 + 3}):
      yield kernel, dilation, length


@pytest.mark.parametrize('kernel,dilation,length', list(cases_1d()))
@pytest.mark.parametrize('use_rpb', [True, False])
def test_na1d_qk_matches_reference(kernel, dilation, length, use_rpb):
  torch.manual_seed(kernel * 1000 + dilation * 37 + length)
  query = torch.randn(2, 3, length, 8, dtype=torch.float64)
  key = torch.randn_like(query)
  rpb = torch.randn(3, 2 * kernel - 1, dtype=torch.float64) if use_rpb else None

  actual = na.na1d_qk_torch(query, key, kernel, dilation, rpb=rpb)
  expected = ref_na1d_qk(query, key, kernel, dilation, rpb=rpb)

  assert actual.shape == (2, 3, length, kernel)
  torch.testing.assert_close(actual, expected, **EXACT)


@pytest.mark.parametrize('kernel,dilation,length', list(cases_1d()))
def test_na1d_av_matches_reference(kernel, dilation, length):
  torch.manual_seed(kernel * 991 + dilation * 13 + length)
  attn = torch.randn(2, 3, length, kernel, dtype=torch.float64).softmax(-1)
  value = torch.randn(2, 3, length, 8, dtype=torch.float64)

  actual = na.na1d_av_torch(attn, value, kernel, dilation)
  expected = ref_na1d_av(attn, value, kernel, dilation)

  assert actual.shape == (2, 3, length, 8)
  torch.testing.assert_close(actual, expected, **EXACT)


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def cases_2d():
  """Kept small: the naive reference is O(X * Y * kernel^2 * dim) in Python."""
  for kernel, dilation in itertools.product([3, 5], [1, 2]):
    window = kernel * dilation
    for size_y in sorted({window, window + 3}):
      yield kernel, dilation, window, size_y


@pytest.mark.parametrize('kernel,dilation,size_x,size_y', list(cases_2d()))
@pytest.mark.parametrize('use_rpb', [True, False])
def test_na2d_qk_matches_reference(kernel, dilation, size_x, size_y, use_rpb):
  torch.manual_seed(kernel * 77 + dilation * 5 + size_y)
  query = torch.randn(2, 3, size_x, size_y, 8, dtype=torch.float64)
  key = torch.randn_like(query)
  rpb = (
    torch.randn(3, 2 * kernel - 1, 2 * kernel - 1, dtype=torch.float64) if use_rpb else None
  )

  actual = na.na2d_qk_torch(query, key, kernel, dilation, rpb=rpb)
  expected = ref_na2d_qk(query, key, kernel, dilation, rpb=rpb)

  assert actual.shape == (2, 3, size_x, size_y, kernel * kernel)
  torch.testing.assert_close(actual, expected, **EXACT)


@pytest.mark.parametrize('kernel,dilation,size_x,size_y', list(cases_2d()))
def test_na2d_av_matches_reference(kernel, dilation, size_x, size_y):
  torch.manual_seed(kernel * 131 + dilation * 17 + size_y)
  attn = torch.randn(2, 3, size_x, size_y, kernel * kernel, dtype=torch.float64).softmax(-1)
  value = torch.randn(2, 3, size_x, size_y, 8, dtype=torch.float64)

  actual = na.na2d_av_torch(attn, value, kernel, dilation)
  expected = ref_na2d_av(attn, value, kernel, dilation)

  torch.testing.assert_close(actual, expected, **EXACT)


# ---------------------------------------------------------------------------
# Against natten itself, where available
# ---------------------------------------------------------------------------

@requires_natten
@pytest.mark.parametrize('kernel,dilation,length', list(cases_1d()))
@pytest.mark.parametrize('use_rpb', [True, False])
def test_na1d_matches_natten(kernel, dilation, length, use_rpb):
  torch.manual_seed(kernel * 17 + dilation * 3 + length)
  query = torch.randn(2, 3, length, 8, dtype=torch.float64)
  key = torch.randn_like(query)
  value = torch.randn_like(query)
  rpb = torch.randn(3, 2 * kernel - 1, dtype=torch.float64) if use_rpb else None

  expected = natten_na1d_qk(query, key, kernel, dilation, rpb=rpb)
  torch.testing.assert_close(
    na.na1d_qk_torch(query, key, kernel, dilation, rpb=rpb), expected, **EXACT
  )

  probs = expected.softmax(-1)
  torch.testing.assert_close(
    na.na1d_av_torch(probs, value, kernel, dilation),
    natten_na1d_av(probs, value, kernel, dilation),
    **EXACT,
  )


@requires_natten
@pytest.mark.parametrize('kernel,dilation,size_x,size_y', list(cases_2d()))
@pytest.mark.parametrize('use_rpb', [True, False])
def test_na2d_matches_natten(kernel, dilation, size_x, size_y, use_rpb):
  torch.manual_seed(kernel * 23 + dilation * 7 + size_y)
  query = torch.randn(2, 3, size_x, size_y, 8, dtype=torch.float64)
  key = torch.randn_like(query)
  value = torch.randn_like(query)
  rpb = (
    torch.randn(3, 2 * kernel - 1, 2 * kernel - 1, dtype=torch.float64) if use_rpb else None
  )

  expected = natten_na2d_qk(query, key, kernel, dilation, rpb=rpb)
  torch.testing.assert_close(
    na.na2d_qk_torch(query, key, kernel, dilation, rpb=rpb), expected, **EXACT
  )

  probs = expected.softmax(-1)
  torch.testing.assert_close(
    na.na2d_av_torch(probs, value, kernel, dilation),
    natten_na2d_av(probs, value, kernel, dilation),
    **EXACT,
  )


# ---------------------------------------------------------------------------
# Wiring and behaviour
# ---------------------------------------------------------------------------

def test_backend_is_reported():
  assert na.BACKEND in ('torch', 'natten')
  if na.BACKEND == 'torch':
    assert na.na1d_qk is na.na1d_qk_torch


def test_allin1_geometry():
  """The exact shapes allin1's encoder produces, in its real dtype.

  1D time attention with the dilation ladder from AllInOneEncoder
  (dilation_factor ** i for i in range(depth=11)), and 2D instrument attention
  over (4 stems padded to kernel_size, frames).
  """
  kernel, heads, dim, stems, frames = 5, 2, 12, 4, 512
  for dilation in [2 ** i for i in range(11)]:
    padded = max(frames, kernel * dilation)
    torch.manual_seed(dilation)
    query = torch.randn(stems, heads, padded, dim)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    rpb = torch.randn(heads, 2 * kernel - 1)

    scores = na.na1d_qk_torch(query, key, kernel, dilation, rpb=rpb)
    assert scores.shape == (stems, heads, padded, kernel)
    assert torch.isfinite(scores).all(), f'dilation={dilation} produced -inf'

    probs = scores.softmax(-1)
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs[..., 0]))
    out = na.na1d_av_torch(probs, value, kernel, dilation)
    assert out.shape == query.shape

  torch.manual_seed(0)
  query = torch.randn(1, heads, kernel, frames, dim)
  key = torch.randn_like(query)
  value = torch.randn_like(query)
  rpb2 = torch.randn(heads, 2 * kernel - 1, 2 * kernel - 1)
  scores = na.na2d_qk_torch(query, key, kernel, 1, rpb=rpb2)
  assert scores.shape == (1, heads, kernel, frames, kernel * kernel)
  assert torch.isfinite(scores).all()
  assert na.na2d_av_torch(scores.softmax(-1), value, kernel, 1).shape == query.shape


def test_gather_chunking_does_not_change_results():
  """The 2D chunking is a memory knob, not a numerics knob."""
  torch.manual_seed(3)
  query = torch.randn(1, 2, 5, 96, 12, dtype=torch.float64)
  key = torch.randn_like(query)
  rpb = torch.randn(2, 9, 9, dtype=torch.float64)

  reference = na.na2d_qk_torch(query, key, 5, 1, rpb=rpb)
  original = na._GATHER_BUDGET
  try:
    for budget in (1, 500, 20_000):
      na.set_gather_budget(budget)
      torch.testing.assert_close(
        na.na2d_qk_torch(query, key, 5, 1, rpb=rpb), reference, **EXACT
      )
  finally:
    na.set_gather_budget(original)


def test_gradients_flow():
  """allin1 only runs inference, but the fallback stays differentiable so the
  training code in allin1/training still works."""
  torch.manual_seed(11)
  query = torch.randn(2, 2, 64, 8, dtype=torch.float64, requires_grad=True)
  key = torch.randn(2, 2, 64, 8, dtype=torch.float64, requires_grad=True)
  rpb = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)

  na.na1d_qk_torch(query, key, 5, 4, rpb=rpb).square().sum().backward()

  for name, tensor in (('query', query), ('key', key), ('rpb', rpb)):
    assert tensor.grad is not None, name
    assert torch.isfinite(tensor.grad).all(), name
    assert tensor.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
  'device',
  [
    pytest.param('cpu', id='cpu'),
    pytest.param(
      'cuda',
      id='cuda',
      marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device'),
    ),
    pytest.param(
      'mps',
      id='mps',
      marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='no MPS device'
      ),
    ),
  ],
)
def test_runs_on_available_devices(device):
  """Plain torch ops, so they follow the tensors -- CUDA (incl. sm_120) and MPS
  need no special build. Checked against CPU."""
  torch.manual_seed(5)
  query = torch.randn(2, 2, 128, 12)
  key = torch.randn_like(query)
  value = torch.randn_like(query)
  rpb = torch.randn(2, 9)

  expected = na.na1d_qk_torch(query, key, 5, 8, rpb=rpb)
  actual = na.na1d_qk_torch(query.to(device), key.to(device), 5, 8, rpb=rpb.to(device))
  torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)

  probs = expected.softmax(-1)
  torch.testing.assert_close(
    na.na1d_av_torch(probs.to(device), value.to(device), 5, 8).cpu(),
    na.na1d_av_torch(probs, value, 5, 8),
    rtol=1e-4,
    atol=1e-5,
  )


def test_unsupported_features_raise():
  """Silently ignoring these would be worse than failing."""
  query = torch.randn(1, 1, 16, 4)
  with pytest.raises(NotImplementedError, match='is_causal'):
    na.na1d_qk_torch(query, query, 5, 1, is_causal=True)
  with pytest.raises(NotImplementedError, match='additional_keys'):
    na.na1d_qk_torch(query, query, 5, 1, additional_keys=query)
