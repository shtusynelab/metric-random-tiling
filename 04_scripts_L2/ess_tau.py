# -*- coding: utf-8 -*-
"""analyze_ess.py -- tau_int / snapshot-independence from ess_pilot series.

Sokal automatic windowing on the logged sum(l) series (dt=100 steps):
  tau_int = 1 + 2 sum_{t=1..W} rho(t), W = smallest window with W >= c*tau_int(W), c = 5
Reported per chain: tau_int (steps), rho at the production thinning lag
(thin = 4e4 steps = 400 samples), the variance-inflation factor of the
20-snapshot production mean (lags k*thin, k = 1..19), and the implied
number of effectively independent snapshots.
usage: python analyze_ess.py <ess_a123.npz> <ess_b12.npz> ...
"""
import sys
import numpy as np


def tau_int_sokal(x, c=5.0):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    # FFT autocorrelation
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:n].real
    acov /= n
    rho = acov / acov[0]
    tau_w = np.empty(n)
    for W in range(1, n):
        tau_w[W] = 1.0 + 2.0 * rho[1:W + 1].sum()
        if W >= c * tau_w[W]:
            W_star = W
            break
    else:
        W_star = n - 1
    return tau_w[W_star], W_star, rho


def main():
    thin_steps = 40_000
    for path in sys.argv[1:]:
        z = np.load(path)
        s = z["series"]
        dt = int(np.asarray(z["dt"]).reshape(-1)[0])
        n = len(s)
        tau, W, rho = tau_int_sokal(s)
        tau_steps = tau * dt
        lag = thin_steps // dt
        rho_thin = rho[lag] if lag < n else float("nan")
        # variance inflation of a 20-snapshot mean spaced `lag` samples apart
        ks = np.arange(1, 20)
        lags = ks * lag
        lags = lags[lags < n]
        infl = 1.0 + 2.0 * sum((1.0 - k / 20.0) * rho[k * lag]
                               for k in range(1, len(lags) + 1))
        n_eff = 20.0 / infl
        err = tau * np.sqrt(2.0 * (2 * W + 1) / n)
        print(f"{path}: n={n} dt={dt}  tau_int={tau_steps:.0f} steps "
              f"(+/-{err * dt:.0f}; W*={W})  mean={s.mean():.4f} sd={s.std():.4f}")
        print(f"   rho(thin={thin_steps}) = {rho_thin:.4f}  "
              f"var-inflation(20 snapshots) = {infl:.3f}  "
              f"n_eff = {n_eff:.1f} / 20")


if __name__ == "__main__":
    main()
