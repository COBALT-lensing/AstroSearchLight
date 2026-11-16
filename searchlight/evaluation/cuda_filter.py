"""Float64, batched FFT evaluation of a common-grid production template bank.

Preparation uses the full resampled profile for polarity and normalization.
Scoring crops only coefficients that cannot overlap an observed sample; this
preserves full-profile normalization and scores. Detection computes the full
correlation envelope, including padded positions, for production peak semantics.
"""
from collections import OrderedDict
import math

import numpy as np


class FFTBank:
    def __init__(self, templates, *, device='cuda', batch_size=256, cache_gib=12, host_cache_gib=32):
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('cuda_fft requires a CUDA GPU visible to PyTorch')
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.cache_limit = int(cache_gib * 1024**3)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.host_cache = OrderedDict()
        self.host_cache_bytes = 0
        self.host_cache_limit = int(host_cache_gib * 1024**3)
        rows = list(templates)
        if not rows:
            raise ValueError('Empty template bank')
        self.time = np.asarray(rows[0].time, float)
        if any(t.time is not rows[0].time and not np.array_equal(t.time, self.time) for t in rows):
            raise ValueError('cuda_fft requires a common template time grid')
        self.raw = torch.as_tensor(np.stack([t.flux for t in rows]), dtype=torch.float64, device=device)
        if not torch.isfinite(self.raw).all():
            raise ValueError("Nonfinite template flux")
        self.half_width = [t.event_half_width for t in rows]
        self.size = len(rows)

    def _grid(self, cadence):
        from .. import matched_filtering as mf
        if np.isclose(mf._median_cadence(self.time, 'template time'), cadence, rtol=1e-6, atol=0):
            grid = self.time
        else:
            grid = np.arange(self.time[0], self.time[-1] + .5 * cadence, cadence)
        # NumPy interp computes slope*(x-x0)+y0, with endpoint clamping.
        lo = np.clip(np.searchsorted(self.time, grid, side='right') - 1, 0, len(self.time)-2)
        delta = grid - self.time[lo]
        width = self.time[lo+1] - self.time[lo]
        torch = self.torch
        args = [torch.as_tensor(x, device=self.device) for x in (lo, delta, width)]
        return grid, args

    def _interpolate(self, start, stop, args):
        lo, delta, width = args
        left, right = self.raw[start:stop, lo], self.raw[start:stop, lo+1]
        values = (right-left) / width * delta + left
        # np.interp clamps past either endpoint.
        values[:, delta < 0] = self.raw[start:stop, :1]
        end = delta > width
        if end.any():
            values[:, end] = self.raw[start:stop, -1:]
        return values

    def _move(self, entry, device):
        return {key: ([x.to(device) for x in value] if key == 'args' else
                      value.to(device) if self.torch.is_tensor(value) else value)
                for key, value in entry.items()}

    def _retain(self, key, entry):
        while self.cache and self.cache_bytes + entry['bytes'] > self.cache_limit:
            old_key, old = self.cache.popitem(last=False)
            self.cache_bytes -= old['bytes']
            while self.host_cache and self.host_cache_bytes + old['bytes'] > self.host_cache_limit:
                _, removed = self.host_cache.popitem(last=False)
                self.host_cache_bytes -= removed['bytes']
            if old['bytes'] <= self.host_cache_limit:
                self.host_cache[old_key] = self._move(old, 'cpu')
                self.host_cache_bytes += old['bytes']
        if entry['bytes'] <= self.cache_limit:
            self.cache[key] = entry
            self.cache_bytes += entry['bytes']
        return entry

    def prepare(self, cadence, radius):
        key = (float(cadence), int(radius))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key in self.host_cache:
            entry = self.host_cache.pop(key)
            self.host_cache_bytes -= entry['bytes']
            return self._retain(key, self._move(entry, self.device))
        torch = self.torch
        grid, args = self._grid(cadence)
        length, center = len(grid), int(np.argmin(np.abs(grid)))
        low, high = max(0, center-radius), min(length, center+radius+1)
        kernels, means, energies, eligible = [], [], [], []
        for start in range(0, self.size, self.batch_size):
            stop = min(start+self.batch_size, self.size)
            values = self._interpolate(start, stop, args)
            if length % 2:
                median = values.median(dim=1).values
            else:
                median = (values.kthvalue(length//2, dim=1).values +
                          values.kthvalue(length//2+1, dim=1).values) / 2
            good = values[:, center] > median
            mean = values.mean(dim=1)
            centered = values - mean[:, None]
            energy = torch.sqrt((centered*centered).sum(dim=1))
            good &= energy > 0
            kernel = centered[:, low:high] / energy.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
            kernels.append(kernel.contiguous())
            means.append(mean); energies.append(energy); eligible.append(good)
        entry = dict(kernels=torch.cat(kernels), means=torch.cat(means), energies=torch.cat(energies),
                     good=torch.cat(eligible), args=args, length=length, center=center, crop_center=center-low)
        entry['bytes'] = (sum(v.numel()*v.element_size() for v in entry.values() if torch.is_tensor(v))
                          + sum(v.numel()*v.element_size() for v in args))
        return self._retain(key, entry)

    def score(self, sample_id, time, flux):
        from .scoring import MatchedFilterScore
        from .. import matched_filtering as mf
        torch = self.torch
        time, flux = np.asarray(time), np.asarray(flux)
        cadence = mf._median_cadence(time, 'time')
        n = len(time)
        bank = self.prepare(cadence, n-1)
        m, center = bank['length'], bank['center']
        positions = np.arange(n) if n <= m else np.arange(center, n-m+center+1)
        k = bank['kernels'].shape[1]
        indices = torch.as_tensor(positions + k-1-bank['crop_center'], device=self.device)
        fft_size = 1 << (n+k-2).bit_length()
        signal_fft = torch.fft.rfft(torch.as_tensor(flux-flux.mean(), device=self.device), n=fft_size)
        best = None
        for start in range(0, self.size, self.batch_size):
            stop = min(start+self.batch_size, self.size)
            kernel_fft = torch.fft.rfft(bank['kernels'][start:stop].flip(1), n=fft_size)
            corr = torch.fft.irfft(kernel_fft*signal_fft, n=fft_size)[:, indices]
            corr[~bank['good'][start:stop]] = -torch.inf
            scores, offset = corr.max(dim=1)
            value, row = scores.max(dim=0)
            value, row = float(value), int(row)
            if math.isfinite(value) and (best is None or value > best.score):
                best = MatchedFilterScore(sample_id, value, float(time[positions[int(offset[row])]]), start+row)
        if best is None:
            raise ValueError('No eligible finite matched-filter score')
        return best

    def detect(self, time, flux, *, height):
        from scipy.signal import find_peaks
        from .. import matched_filtering as mf
        torch = self.torch
        time, flux = np.asarray(time), np.asarray(flux)
        cadence = mf._median_cadence(time, 'time')
        n = len(time)
        polarity_flux = flux-np.median(flux)
        min_center = mf._minimum_center_flux(polarity_flux)
        if np.ptp(flux) == 0 or np.max(polarity_flux) < min_center:
            return []
        if self.score('event-precheck', time, flux).score < height - 1e-10*max(1., abs(height)):
            return []
        bank = self.prepare(cadence, n-1)
        m, center = bank['length'], bank['center']
        full_length = n+m-1
        fft_size = 1 << (full_length-1).bit_length()
        signal_fft = torch.fft.rfft(torch.as_tensor(flux-flux.mean(), device=self.device), n=fft_size)
        envelope = torch.full((full_length,), -torch.inf, device=self.device, dtype=torch.float64)
        winner = torch.full((full_length,), -1, device=self.device, dtype=torch.long)
        for start in range(0, self.size, self.batch_size):
            stop = min(start+self.batch_size, self.size)
            values = self._interpolate(start, stop, bank['args'])
            values = (values-bank['means'][start:stop, None]) / bank['energies'][start:stop, None].clamp_min(torch.finfo(torch.float64).tiny)
            corr = torch.fft.irfft(torch.fft.rfft(values.flip(1), n=fft_size)*signal_fft, n=fft_size)[:, :full_length]
            corr[~bank['good'][start:stop]] = -torch.inf
            scores, rows = corr.max(dim=0)
            improved = scores > envelope
            envelope[improved] = scores[improved]
            winner[improved] = start+rows[improved]
        envelope, winner = envelope.cpu().numpy(), winner.cpu().numpy()
        if n <= m:
            # The production padding is m samples on either side: its valid
            # correlation is the full unpadded correlation with one zero each end.
            envelope = np.r_[0., envelope, 0.]
            winner = np.r_[-1, winner, -1]
            center_shift = center-m
        else:
            envelope, winner = envelope[m-1:n], winner[m-1:n]
            center_shift = center
        peaks, props = find_peaks(envelope, height=height, prominence=0)
        min_center = mf._minimum_center_flux(flux-np.median(flux))
        matches = []
        for peak, prominence in zip(peaks, props['prominences']):
            observed = int(peak)+center_shift
            if not 0 <= observed < n or flux[observed]-np.median(flux) < min_center:
                continue
            index = int(winner[peak])
            if index < 0:
                continue
            half = self.half_width[index]
            if half is None or not np.isfinite(half) or half <= 0:
                half = .5*(m-1)*cadence
            peak_time = float(time[observed])
            low, high = max(float(time[0]), peak_time-half), min(float(time[-1]), peak_time+half)
            matches.append(mf.MatchedFilterMatch(index, observed,
                max(0, int(np.searchsorted(time, low, side='left'))),
                min(n-1, int(np.searchsorted(time, high, side='right')-1)),
                peak_time, low, high, float(envelope[peak]), float(prominence)))
        return mf._finalize_matches(matches, dedupe_overlaps=True, max_matches=None)


_ENGINE = None
_ENGINE_TEMPLATES = None


def engine(templates):
    global _ENGINE, _ENGINE_TEMPLATES
    if templates is not _ENGINE_TEMPLATES:
        _ENGINE = FFTBank(templates)
        _ENGINE_TEMPLATES = templates
    return _ENGINE
