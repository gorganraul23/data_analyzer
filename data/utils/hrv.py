import math
import numpy as np

import matplotlib
matplotlib.use("Agg")

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl

############################################################### pyhrv lib

def compute_rmssd_lib(rr_list):
    results = td.rmssd(rr_list)
    return results['rmssd']

def compute_sdnn_lib(rr_list):
    results = td.sdnn(rr_list)
    return results['sdnn']

def compute_hr_mean_std_lib(rr_list):
    results = td.hr_parameters(rr_list)
    return {
        'mean': results['hr_mean'],
        'std': results['hr_std'],
    }

def compute_rr_mean_lib(rr_list):
    results = td.nni_parameters(rr_list)
    return {
        'mean': results['nni_mean'],
        'std': abs(results['nni_max'] - results['nni_min']),
    }

def compute_nn50_lib(rr_list):
    results = td.nn50(rr_list)
    return results['nn50']

def compute_pnn50_lib(rr_list):
    results = td.time_domain(rr_list, plot=False, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return results['pnn50']

def compute_tinn_lib(rr_list):
    results = td.tinn(rr_list, plot=False, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return results['tinn']

def compute_ftt_lib(rr_list):
    results = fd.welch_psd(nni=rr_list, show=False)
    if "plot" in results:
        results.pop("plot", None)
    vlf, lf, hf = results['fft_abs']
    return {
        'lf': float(lf),
        'hf': float(hf),
        'ratio': float(results['fft_ratio']),
    }

def compute_sd_lib(rr_list):
    results = nl.poincare(rr_list, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return {
        'sd1': results['sd1'],
        'sd2': results['sd2'],
        'ratio': results['sd_ratio'],
    }

def compute_sampen_lib(rr_list):
    results = nl.sample_entropy(rr_list)
    return results['sampen']

def compute_dfa_lib(rr_list):
    results = nl.dfa(rr_list, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return {
        'dfa_a1': results['dfa_alpha1'],
        'dfa_a2': results['dfa_alpha2'],
    }

############################################################### manual formulas

# --------- Helpers ---------

def clean_rr_list(ibi_ms):
    rr = np.asarray(ibi_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0.0]
    rr = rr[(rr >= 300.0) & (rr <= 2000.0)]

    return rr

def _safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")

# --------- Helpers for SD computation ---------

try:
    from numpy.lib.stride_tricks import sliding_window_view
    def _embed(x, m):
        x = np.asarray(x, dtype=float)
        if x.size <= m:
            return None
        return sliding_window_view(x, m)  # shape: (N-m+1, m)
except Exception:
    def _embed(x, m):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n <= m:
            return None
        return np.array([x[i:i+m] for i in range(n - m + 1)])

# --------- Helpers for frequency domain ---------

try:
    from scipy.interpolate import interp1d as _interp1d
except Exception:
    _interp1d = None

try:
    from scipy.signal import welch as _scipy_welch
except Exception:
    _scipy_welch = None

def _tol(x, r):
    if r is not None:
        return float(r)
    if x.size < 2:
        return float("nan")
    s = np.std(x, ddof=1)
    return float(0.2 * s) if s > 0 else float("nan")

# --------- Parameters computing ---------

# --------- Time domain ---------

def compute_mean_hr(rr_list):
    return 60 * 1000 / np.mean(rr_list)

def compute_mean_rr(rr_list):
    return np.mean(rr_list)

def compute_sdnn(rr_list):
    return float(np.std(rr_list, ddof=1)) if rr_list.size > 1 else float("nan")

def compute_rmssd(rr_list):
    if rr_list.size < 2:
        return float("nan")

    return np.sqrt(np.mean(np.square(np.diff(rr_list))))

def compute_nn50(rr_list, threshold_ms=50.0):
    if rr_list.size < 2:
        return float("nan")

    return np.sum(np.abs(np.diff(rr_list)) > threshold_ms)

def compute_pnn50(rr_list, threshold_ms=50.0):
    if rr_list.size < 2:
        return float("nan")

    return 100 * compute_nn50(rr_list, threshold_ms) / len(rr_list)

def compute_tinn(rr_list, bin_ms=7.8125, peak_frac=0.05):

    if rr_list.size < 3:
        return float("nan")
    rmin, rmax = float(np.min(rr_list)), float(np.max(rr_list))
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
        return float("nan")

    # Build histogram
    bins = np.arange(rmin, rmax + bin_ms, bin_ms)
    if bins.size < 4:
        return float("nan")
    hist, edges = np.histogram(rr_list, bins=bins)
    if hist.size < 3 or hist.sum() == 0:
        return float("nan")

    # Find peak and threshold
    peak_idx = int(np.argmax(hist))
    thr = float(peak_frac) * hist[peak_idx]

    # march left/right to ~zero
    left = peak_idx
    while left > 0 and hist[left] > thr:
        left -= 1
    right = peak_idx
    while right < hist.size - 1 and hist[right] > thr:
        right += 1

    left_edge = edges[max(left, 0)]
    right_edge = edges[min(right, len(edges) - 1)]
    width = right_edge - left_edge

    return float(width if width > 0 else float("nan"))


# --------- SD ---------
def compute_sd1(rr_list):
    if rr_list.size < 2:
        return float("nan")
    u = (rr_list[1:] - rr_list[:-1]) / math.sqrt(2.0)
    return float(np.std(u, ddof=1)) if u.size > 1 else float("nan")

def compute_sd2(rr_list):
    if rr_list.size < 2:
        return float("nan")
    v = (rr_list[1:] + rr_list[:-1]) / math.sqrt(2.0)
    return float(np.std(v, ddof=1)) if v.size > 1 else float("nan")

def compute_sd2_sd1(rr_list):
    _sd1 = compute_sd1(rr_list)
    _sd2 = compute_sd2(rr_list)
    return float(_sd2 / _sd1) if (_sd1 and not math.isnan(_sd1) and _sd1 != 0 and not math.isnan(_sd2)) else float("nan")

def compute_sd1_from_rmssd(rr_list):
    r = compute_rmssd(rr_list)
    return float(r / math.sqrt(2.0)) if not math.isnan(r) else float("nan")

def compute_sd2_from_sdnn_rmssd(rr_list):
    s = compute_sdnn(rr_list); r = compute_rmssd(rr_list)
    if math.isnan(s) or math.isnan(r): return float("nan")
    return float(math.sqrt(max(0.0, 2.0 * (s**2) - 0.5 * (r**2))))

def compute_sd2_sd1_from_rmssd_sdnn(rr_list):
    _sd1 = compute_sd1(rr_list)
    _sd2 = compute_sd2(rr_list)
    return float(_sd2 / _sd1) if (_sd1 and not math.isnan(_sd1) and _sd1 != 0 and not math.isnan(_sd2)) else float("nan")

# ----- Approximate Entropy (ApEn) -----

def compute_ap_en(rr_list, m=2, r=None):

    if rr_list.size <= m + 1:
        return float("nan")

    r = _tol(rr_list, r)
    if not np.isfinite(r) or r <= 0:
        return float("nan")

    emb_m  = _embed(rr_list, m)
    emb_m1 = _embed(rr_list, m + 1)
    if emb_m is None or emb_m1 is None:
        return float("nan")

    # φ(m): for each template, fraction of matches (including self)
    # Distance matrix with Chebyshev norm
    dist_m = np.max(np.abs(emb_m[:, None, :] - emb_m[None, :, :]), axis=2)
    Cm = np.mean(dist_m <= r, axis=1)
    phi_m = float(np.mean(Cm)) if np.isfinite(Cm).all() else float("nan")

    # φ(m+1)
    dist_m1 = np.max(np.abs(emb_m1[:, None, :] - emb_m1[None, :, :]), axis=2)
    Cm1 = np.mean(dist_m1 <= r, axis=1)
    phi_m1 = float(np.mean(Cm1)) if np.isfinite(Cm1).all() else float("nan")

    if phi_m1 <= 0 or not np.isfinite(phi_m1) or phi_m <= 0 or not np.isfinite(phi_m):
        return float("nan")

    # ApEn = ln(phi_m / phi_m1)  (equivalently: -ln(phi_m1/phi_m))
    return float(np.log(phi_m / phi_m1))

# ----- Sample Entropy (SampEn) -----

def compute_samp_en(rr_list, m=2, r=None):

    if rr_list.size <= m + 1:
        return float("nan")

    r = _tol(rr_list, r)
    if not np.isfinite(r) or r <= 0:
        return float("nan")

    emb_m  = _embed(rr_list, m)
    emb_m1 = _embed(rr_list, m + 1)
    if emb_m is None or emb_m1 is None:
        return float("nan")

    # Distances (Chebyshev)
    Dm  = np.max(np.abs(emb_m[:,  None, :] - emb_m[None,  :,  :]), axis=2)
    Dm1 = np.max(np.abs(emb_m1[:, None, :] - emb_m1[None, :,  :]), axis=2)

    # Exclude self-matches via strictly upper triangle (i<j)
    M  = Dm.shape[0]
    M1 = Dm1.shape[0]
    iu  = np.triu_indices(M,  k=1)
    iu1 = np.triu_indices(M1, k=1)

    B = int(np.sum(Dm[iu]  <= r))   # matches for m
    A = int(np.sum(Dm1[iu1] <= r))  # matches for m+1

    if B == 0 or A == 0:
        return float("nan")

    return float(-np.log(A / B))

# ----- DFA alpha1 and alpha2 -----

def dfa_alpha(rr_list, scales, bidirectional=True):

    n = rr_list.size
    if n < (max(scales) if scales else 0):
        return float("nan")

    # Integrated (profile) signal
    x = rr_list - np.mean(rr_list)
    y = np.cumsum(x - np.mean(x))

    Fs = []
    Ss = []

    for s in scales:
        s = int(s)
        if s < 2 or n < s:
            continue

        # forward windows
        Ns_f = n // s
        if Ns_f >= 2:
            rms_list = []
            for k in range(Ns_f):
                seg = y[k*s:(k+1)*s]
                t = np.arange(s, dtype=float)
                # linear detrend
                p = np.polyfit(t, seg, 1)
                trend = np.polyval(p, t)
                rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))

            # optional: add reverse windows from the end to use leftover samples
            if bidirectional:
                rms_list_rev = []
                for k in range(Ns_f):
                    seg = y[n - (k+1)*s : n - k*s]
                    if seg.size != s: break
                    t = np.arange(s, dtype=float)
                    p = np.polyfit(t, seg, 1)
                    trend = np.polyval(p, t)
                    rms_list_rev.append(np.sqrt(np.mean((seg - trend) ** 2)))
                rms_list.extend(rms_list_rev)

            if rms_list:
                F_s = float(np.mean(rms_list))
                if np.isfinite(F_s) and F_s > 0:
                    Fs.append(F_s)
                    Ss.append(float(s))

    if len(Fs) < 2:
        return float("nan")

    logF = np.log(Fs)
    logS = np.log(Ss)
    alpha, _ = np.polyfit(logS, logF, 1)
    return float(alpha)

def compute_dfa_a1(rr_list):
    return dfa_alpha(rr_list, scales=range(4, 17), bidirectional=True)

def compute_dfa_a2(rr_list):
    return dfa_alpha(rr_list, scales=range(16, 65), bidirectional=True)

# ----- Stress Index -----

def _hist_edges_ms(rr, bin_ms):
    rmin, rmax = float(np.min(rr)), float(np.max(rr))
    left  = bin_ms * math.floor(rmin / bin_ms)
    right = bin_ms * math.ceil(rmax / bin_ms)
    edges = np.arange(left, right + bin_ms, bin_ms, dtype=float)
    if edges.size < 2: edges = np.array([rmin, rmin + bin_ms], float)
    return edges

def baevsky_stress_index(rr_list, bin_ms=50.0):
    if rr_list.size < 2: return float('nan')
    edges = _hist_edges_ms(rr_list, bin_ms)
    hist, edges = np.histogram(rr_list, bins=edges)
    if hist.sum() == 0: return float('nan')

    peak = int(np.argmax(hist))
    Amo_pct = 100.0 * hist[peak] / float(rr_list.size)
    Mo_ms   = 0.5 * (edges[peak] + edges[peak+1])
    rng_ms  = float(np.max(rr_list) - np.min(rr_list))
    Mo_s, rng_s = Mo_ms/1000.0, rng_ms/1000.0

    denom = 2.0 * Mo_s * rng_s
    return float(Amo_pct / denom) if denom > 0 else float('nan')

def compute_stress_index(rr_ms, bin_ms=50.0):
    si = baevsky_stress_index(rr_ms, bin_ms=bin_ms)
    return math.sqrt(si) if si > 0 else float('nan')

# --------- PNS SNS Index ---------

def _z(x, mu, sigma):
    return (x - mu) / sigma if (x is not None and math.isfinite(x) and sigma and math.isfinite(sigma)) else float("nan")

def compute_pns_sns(rr_list, norms=None, weights=None):
    if rr_list.size < 3:
        return float("nan"), float("nan")

    mean_rr = compute_mean_rr(rr_list)
    mean_hr = compute_mean_hr(rr_list)
    sd1 = compute_sd1(rr_list)
    sd2 = compute_sd2(rr_list)
    rmssd = compute_rmssd(rr_list)
    stress = compute_stress_index(rr_list)

    # normalize SD1, SD2 by mean RR to get SD1(%), SD2(%)
    sd1_pct = float(100.0 * sd1 / mean_rr) if (math.isfinite(sd1) and mean_rr > 0) else float("nan")
    sd2_pct = float(100.0 * sd2 / mean_rr) if (math.isfinite(sd2) and mean_rr > 0) else float("nan")

    z_rr     = _z(mean_rr, norms['mean_rr']['mu'],  norms['mean_rr']['sd'])
    z_rmssd  = _z(rmssd,   norms['rmssd']['mu'],    norms['rmssd']['sd'])
    z_sd1pct = _z(sd1_pct, norms['sd1_pct']['mu'],  norms['sd1_pct']['sd'])

    z_hr     = _z(mean_hr, norms['mean_hr']['mu'],  norms['mean_hr']['sd'])
    z_stress = _z(stress,  norms['stress']['mu'],   norms['stress']['sd'])
    z_sd2pct = _z(sd2_pct, norms['sd2_pct']['mu'],  norms['sd2_pct']['sd'])

    W = weights or {
        'pns': {'rr': 1/3, 'rmssd': 1/3, 'sd1pct': 1/3},
        'sns': {'hr': 1/3, 'stress': 1/3, 'sd2pct': 1/3},
    }

    pns = (W['pns']['rr']*z_rr + W['pns']['rmssd']*z_rmssd + W['pns']['sd1pct']*z_sd1pct)
    sns = (W['sns']['hr']*z_hr + W['sns']['stress']*z_stress + W['sns']['sd2pct']*z_sd2pct)
    return float(pns), float(sns)


# --------- Frequency domain ---------

def resample_rr_tachogram_ms(rr_list, fs=4.0, kind="cubic"):
    if rr_list.size < 3:
        return None, None

    # cumulative time in seconds
    t = np.cumsum(rr_list) / 1000.0
    t -= t[0]
    if t[-1] <= 1.0:  # too short
        return None, None

    # uniform grid
    dt = 1.0 / float(fs)
    tu = np.arange(dt, t[-1], dt)
    if tu.size < 16:
        return None, None

    # interpolate RR (keep units in ms to match "ms²" band powers)
    if _interp1d is not None:
        f = _interp1d(t, rr_list, kind=kind, fill_value="extrapolate", bounds_error=False)
        rr_u = f(tu)
    else:
        # linear fallback
        rr_u = np.interp(tu, t, rr_list)

    return tu, rr_u

def welch_psd_ms(rr_uniform_ms, fs=4.0, nperseg=256, noverlap=128):
    x = np.asarray(rr_uniform_ms, dtype=float)
    n = x.size
    if n < 16:
        return None, None

    # SciPy preferred
    if _scipy_welch is not None:
        nperseg = min(nperseg, n)
        if nperseg < 16:
            return None, None
        noverlap = min(noverlap, nperseg // 2)
        f, p = _scipy_welch(x, fs=fs, window="hann", nperseg=nperseg,
                            noverlap=noverlap, detrend="constant", scaling="density")
        return f.astype(float), p.astype(float)

    # NumPy fallback (manual Welch)
    nperseg = min(nperseg, n)
    if nperseg < 16:
        return None, None
    noverlap = min(noverlap, nperseg // 2)
    step = nperseg - noverlap
    win = np.hanning(nperseg)
    scale = (np.sum(win**2) * fs)
    segs = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - np.mean(seg)
        X = np.fft.rfft(seg * win)
        Pxx = (np.abs(X) ** 2) / scale
        segs.append(Pxx)
    if not segs:
        return None, None
    Pxx = np.mean(segs, axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, Pxx

def _band_power(f, p, f1, f2):
    mask = (f >= f1) & (f < f2)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(p[mask], f[mask]))

def compute_frequency_domain_metrics(rr_list, fs=4.0,
                           vlf_band=(0.00, 0.04),
                           lf_band=(0.04, 0.15),
                           hf_band=(0.15, 0.40)):
    tu, rr_u = resample_rr_tachogram_ms(rr_list, fs=fs)
    if rr_u is None:
        return {"vlf": float("nan"), "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan")}

    f, p = welch_psd_ms(rr_u, fs=fs)
    if f is None:
        return {"vlf": float("nan"), "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan")}

    vlf = _band_power(f, p, *vlf_band)
    lf  = _band_power(f, p, *lf_band)
    hf  = _band_power(f, p, *hf_band)
    lf_hf = (lf / hf) if (hf and not math.isnan(hf) and hf > 0) else float("nan")

    return {"vlf": vlf, "lf": lf, "hf": hf, "lf_hf": lf_hf}


# --------- Main ---------

def compute_metrics_from_ibi_list(ibi_list):

    rr = clean_rr_list(ibi_list)

    if rr.size < 3:
        return {
            "mean_hr": float("nan"), "mean_rr": float("nan"), "rmssd": float("nan"), "sdnn": float("nan"), "stress_index": float("nan"),
            "pns_index": float("nan"), "sns_index": float("nan"), "nn50": float("nan"), "pnn50": float("nan"), "tinn": float("nan"),
            "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan"),
            "sd1": float("nan"), "sd2": float("nan"), "sd2_sd1": float("nan"),
            "ap_en": float("nan"), "samp_en": float("nan"), "dfa_a1": float("nan"), "dfa_a2": float("nan"),
        }

    mean_hr = compute_mean_hr(rr)
    mean_rr = compute_mean_rr(rr)
    rmssd = compute_rmssd(rr)
    sdnn = compute_sdnn(rr)
    nn50 = compute_nn50(rr)
    pnn50 = compute_pnn50(rr)
    tinn = compute_tinn(rr)

    stress_index = compute_stress_index(rr)

    freq = compute_frequency_domain_metrics(rr, fs=4.0)
    lf_power = freq["lf"]
    hf_power = freq["hf"]
    lf_hf = freq["lf_hf"]

    sd1 = compute_sd1(rr)
    sd2 = compute_sd2(rr)
    sd2_sd1 = compute_sd2_sd1(rr)
    # sd1 = compute_sd1_from_rmssd(rr)
    # sd2 = compute_sd2_from_sdnn_rmssd(rr)
    # sd2_sd1 = compute_sd2_sd1_from_rmssd_sdnn(rr)

    hr_results = compute_hr_mean_std_lib(rr)
    mean_hr_lib = hr_results["mean"]
    std_hr_lib = hr_results["std"]
    rr_results = compute_rr_mean_lib(rr)
    mean_rr_lib = rr_results["mean"]
    std_rr_lib = rr_results["std"]
    norms = {
        'mean_rr': {'mu': mean_rr_lib, 'sd': std_rr_lib},
        'rmssd': {'mu': rmssd, 'sd': 60.5},
        'sd1_pct': {'mu': sd1, 'sd': 20.0},
        'mean_hr': {'mu': mean_hr_lib, 'sd': std_hr_lib},
        'stress': {'mu': stress_index, 'sd': 2},
        'sd2_pct': {'mu': sd2, 'sd': 40.0},
    }
    pns_index, sns_index = compute_pns_sns(rr_list=rr, norms=norms)

    ap_en = compute_ap_en(rr, m=2)
    samp_en = compute_samp_en(rr, m=2)

    dfa_a1 = compute_dfa_a1(rr)
    dfa_a2 = compute_dfa_a2(rr)

    return {
        "mean_hr": mean_hr,
        "mean_rr": mean_rr,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "stress_index": stress_index,
        "pns_index": pns_index,
        "sns_index": sns_index,
        "nn50": nn50,
        "pnn50": pnn50,
        "tinn": tinn,
        "lf": lf_power,
        "hf": hf_power,
        "lf_hf": lf_hf,
        "sd1": sd1,
        "sd2": sd2,
        "sd2_sd1": sd2_sd1,
        "ap_en": ap_en,
        "samp_en": samp_en,
        "dfa_a1": dfa_a1,
        "dfa_a2": dfa_a2,
    }

def compute_metrics_from_ibi_list_lib(ibi_list):

    rr = clean_rr_list(ibi_list)

    if rr.size < 3:
        return {
            "mean_hr": float("nan"), "mean_rr": float("nan"), "rmssd": float("nan"), "sdnn": float("nan"), "stress_index": float("nan"),
            "pns_index": float("nan"), "sns_index": float("nan"), "nn50": float("nan"), "pnn50": float("nan"), "tinn": float("nan"),
            "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan"),
            "sd1": float("nan"), "sd2": float("nan"), "sd2_sd1": float("nan"),
            "ap_en": float("nan"), "samp_en": float("nan"), "dfa_a1": float("nan"), "dfa_a2": float("nan"),
        }

    rmssd = compute_rmssd_lib(rr)
    sdnn = compute_sdnn_lib(rr)
    tinn = compute_tinn_lib(rr)
    nn50 = compute_nn50_lib(rr)
    pnn50 = compute_pnn50_lib(rr)

    hr_results = compute_hr_mean_std_lib(rr)
    mean_hr = hr_results["mean"]
    std_hr = hr_results["std"]

    rr_results = compute_rr_mean_lib(rr)
    mean_rr = rr_results["mean"]
    std_rr = rr_results["std"]

    stress_index = 0
    pns_index, sns_index = 0, 0

    ftt_results = compute_ftt_lib(rr)
    lf_power = ftt_results["lf"]
    hf_power = ftt_results["hf"]
    lf_hf = ftt_results["ratio"]

    sd_results = compute_sd_lib(rr)
    sd1 = sd_results["sd1"]
    sd2 = sd_results["sd2"]
    sd2_sd1 = sd_results["ratio"]

    ap_en = 0
    samp_en = compute_sampen_lib(rr)

    dfa_results = compute_dfa_lib(rr)
    dfa_a1 = dfa_results["dfa_a1"]
    dfa_a2 = dfa_results["dfa_a2"]

    return {
        "mean_hr": mean_hr,
        "mean_rr": mean_rr,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "stress_index": stress_index,
        "pns_index": pns_index,
        "sns_index": sns_index,
        "nn50": nn50,
        "pnn50": pnn50,
        "tinn": tinn,
        "lf": lf_power,
        "hf": hf_power,
        "lf_hf": lf_hf,
        "sd1": sd1,
        "sd2": sd2,
        "sd2_sd1": sd2_sd1,
        "ap_en": ap_en,
        "samp_en": samp_en,
        "dfa_a1": dfa_a1,
        "dfa_a2": dfa_a2,
    }
