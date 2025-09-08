import math
import numpy as np

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl

#<editor-fold desc="PyHRV Lib">

############################################################### Computing using PyHRV Lib

# --------- RMSSD ---------
def compute_rmssd_lib(rr_list):
    results = td.rmssd(rr_list)
    return results['rmssd']

# --------- SDNN ---------
def compute_sdnn_lib(rr_list):
    results = td.sdnn(rr_list)
    return results['sdnn']

# --------- Mean HR ---------
def compute_hr_mean_std_lib(rr_list):
    results = td.hr_parameters(rr_list)
    return {
        'mean': results['hr_mean'],
        'std': results['hr_std'],
    }

# --------- Mean RR ---------
def compute_rr_mean_lib(rr_list):
    results = td.nni_parameters(rr_list)
    return {
        'mean': results['nni_mean'],
        'std': abs(results['nni_max'] - results['nni_min']),
    }

# --------- NN50 ---------
def compute_nn50_lib(rr_list):
    results = td.nn50(rr_list)
    return results['nn50']

# --------- pNN50 ---------
def compute_pnn50_lib(rr_list):
    results = td.time_domain(rr_list, plot=False, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return results['pnn50']

# --------- TINN ---------
def compute_tinn_lib(rr_list):
    results = td.tinn(rr_list, plot=False, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return results['tinn']

# --------- Freq domain ---------
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

# --------- SD ---------
def compute_sd_lib(rr_list):
    results = nl.poincare(rr_list, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return {
        'sd1': results['sd1'],
        'sd2': results['sd2'],
        'ratio': results['sd_ratio'],
    }

# --------- SampEn ---------
def compute_sampen_lib(rr_list):
    results = nl.sample_entropy(rr_list)
    return results['sampen']

# --------- DFA ---------
def compute_dfa_lib(rr_list):
    results = nl.dfa(rr_list, show=False)
    if "plot" in results:
        results.pop("plot", None)
    return {
        'dfa_a1': results['dfa_alpha1'],
        'dfa_a2': results['dfa_alpha2'],
    }

#</editor-fold>

#<editor-fold desc="Chunk computing">

############################################################### Computing using chunks

def split_into_chunks(ibi_df: pd.DataFrame):

    if ibi_df is None or len(ibi_df) < 2:
        return []

    if not {"id", "value_ibi"}.issubset(ibi_df.columns):
        raise ValueError("ibi_df must contain 'id' and 'value_ibi' columns")

    ibi_df = ibi_df.sort_values("id").reset_index(drop=True)

    chunks = []
    current_chunk = [ibi_df.iloc[0]["value_ibi"]]

    for i in range(1, len(ibi_df)):
        prev_id = ibi_df.iloc[i - 1]["id"]
        curr_id = ibi_df.iloc[i]["id"]

        if curr_id - prev_id <= 1:
            current_chunk.append(ibi_df.iloc[i]["value_ibi"])
        else:
            if len(current_chunk) >= 2:
                chunks.append(current_chunk)
            current_chunk = [ibi_df.iloc[i]["value_ibi"]]

    if len(current_chunk) >= 2:
        chunks.append(current_chunk)

    return chunks

# --------- RMSSD using chunks ---------
def compute_rmssd_chunks_lib(chunks):
    if not chunks:
        return float("nan")
    rmssd_vals = []
    for chunk in chunks:
        result = compute_rmssd_lib(chunk)
        rmssd_vals.append(result)

    return float(np.mean(rmssd_vals)) if rmssd_vals else float("nan")

# --------- SDNN using chunks ---------
def compute_sdnn_chunks_lib(chunks):
    if not chunks:
        return float("nan")
    sdnn_vals = []
    for chunk in chunks:
        result = compute_sdnn_lib(chunk)
        sdnn_vals.append(result)

    return float(np.mean(sdnn_vals)) if sdnn_vals else float("nan")

# --------- NN50 using chunks ---------
def compute_nn50_chunks_lib(chunks):
    if not chunks:
        return float("nan")
    nn50_vals = []
    for chunk in chunks:
        result = compute_nn50_lib(chunk)
        nn50_vals.append(result)

    return float(np.sum(nn50_vals)) if nn50_vals else float("nan")

# --------- pNN50 using chunks ---------
def compute_pnn50_chunks_lib(chunks):
    if not chunks:
        return float("nan")
    pnn50_vals = []
    for chunk in chunks:
        result = compute_pnn50_lib(chunk)
        pnn50_vals.append(result)

    return float(np.mean(pnn50_vals)) if pnn50_vals else float("nan")

# --------- TINN using chunks ---------
def compute_tinn_chunks_lib(chunks):
    if not chunks:
        return float("nan")
    tinn_vals = []
    for chunk in chunks:
        result = compute_tinn_lib(chunk)
        tinn_vals.append(result)

    return float(np.mean(tinn_vals)) if tinn_vals else float("nan")

#</editor-fold>

#<editor-fold desc="Manual formulas">

############################################################### manual formulas

# --------- Helpers ---------
def clean_rr_list(ibi_ms):
    rr = np.asarray(ibi_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0.0]
    # rr = rr[(rr >= 300.0) & (rr <= 2000.0)]

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

######################################### Time domain

# --------- Manual Mean HR using RR list ---------
def compute_mean_hr(rr_list):
    return 60 * 1000 / np.mean(rr_list)

# --------- Manual Mean RR ---------
def compute_mean_rr(rr_list):
    return np.mean(rr_list)

# --------- Manual SDNN ---------
def compute_sdnn(rr_list):
    return float(np.std(rr_list, ddof=1)) if rr_list.size > 1 else float("nan")

# --------- Manual RMSSD ---------
def compute_rmssd(rr_list):
    if rr_list.size < 2:
        return float("nan")
    return np.sqrt(np.mean(np.square(np.diff(rr_list))))

# --------- Manual NN50 ---------
def compute_nn50(rr_list, threshold_ms=50.0):
    if rr_list.size < 2:
        return float("nan")
    return np.sum(np.abs(np.diff(rr_list)) > threshold_ms)

# --------- Manual pNN50 ---------
def compute_pnn50(rr_list, threshold_ms=50.0):
    if rr_list.size < 2:
        return float("nan")
    return 100 * compute_nn50(rr_list, threshold_ms) / len(rr_list)

# --------- Manual TINN ---------
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

# --------- Manual SD1 ---------
def compute_sd1(rr_list):
    if rr_list.size < 2:
        return float("nan")
    u = (rr_list[1:] - rr_list[:-1]) / math.sqrt(2.0)
    return float(np.std(u, ddof=1)) if u.size > 1 else float("nan")

# --------- Manual SD2 ---------
def compute_sd2(rr_list):
    if rr_list.size < 2:
        return float("nan")
    v = (rr_list[1:] + rr_list[:-1]) / math.sqrt(2.0)
    return float(np.std(v, ddof=1)) if v.size > 1 else float("nan")

# --------- Manual SD1/SD1 ---------
def compute_sd2_sd1(rr_list):
    _sd1 = compute_sd1(rr_list)
    _sd2 = compute_sd2(rr_list)
    return float(_sd2 / _sd1) if (_sd1 and not math.isnan(_sd1) and _sd1 != 0 and not math.isnan(_sd2)) else float("nan")

# --------- Manual SD1 from RMSSD ---------
def compute_sd1_from_rmssd(rr_list):
    r = compute_rmssd(rr_list)
    return float(r / math.sqrt(2.0)) if not math.isnan(r) else float("nan")

# --------- Manual SD2 from RMSSD and SDNN ---------
def compute_sd2_from_sdnn_rmssd(rr_list):
    s = compute_sdnn(rr_list); r = compute_rmssd(rr_list)
    if math.isnan(s) or math.isnan(r): return float("nan")
    return float(math.sqrt(max(0.0, 2.0 * (s**2) - 0.5 * (r**2))))

# --------- Manual SD2/SD1 from RMSSD and SDNN ---------
def compute_sd2_sd1_from_rmssd_sdnn(rr_list):
    _sd1 = compute_sd1(rr_list)
    _sd2 = compute_sd2(rr_list)
    return float(_sd2 / _sd1) if (_sd1 and not math.isnan(_sd1) and _sd1 != 0 and not math.isnan(_sd2)) else float("nan")

# ----- Manual Approximate Entropy (ApEn) -----
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

    dist_m = np.max(np.abs(emb_m[:, None, :] - emb_m[None, :, :]), axis=2)
    Cm = np.mean(dist_m <= r, axis=1)
    phi_m = float(np.mean(Cm)) if np.isfinite(Cm).all() else float("nan")

    dist_m1 = np.max(np.abs(emb_m1[:, None, :] - emb_m1[None, :, :]), axis=2)
    Cm1 = np.mean(dist_m1 <= r, axis=1)
    phi_m1 = float(np.mean(Cm1)) if np.isfinite(Cm1).all() else float("nan")

    if phi_m1 <= 0 or not np.isfinite(phi_m1) or phi_m <= 0 or not np.isfinite(phi_m):
        return float("nan")

    return float(np.log(phi_m / phi_m1))

# ----- Manual Sample Entropy (SampEn) -----
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

    Dm  = np.max(np.abs(emb_m[:,  None, :] - emb_m[None,  :,  :]), axis=2)
    Dm1 = np.max(np.abs(emb_m1[:, None, :] - emb_m1[None, :,  :]), axis=2)

    M  = Dm.shape[0]
    M1 = Dm1.shape[0]
    iu  = np.triu_indices(M,  k=1)
    iu1 = np.triu_indices(M1, k=1)

    B = int(np.sum(Dm[iu]  <= r))   # matches for m
    A = int(np.sum(Dm1[iu1] <= r))  # matches for m+1

    if B == 0 or A == 0:
        return float("nan")

    return float(-np.log(A / B))

# ----- Manual DFA -----
def dfa_alpha(rr_list, scales, bidirectional=True):
    n = rr_list.size
    if n < (max(scales) if scales else 0):
        return float("nan")

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

# --------- Manual DFA alpha1 ---------
def compute_dfa_a1(rr_list):
    return dfa_alpha(rr_list, scales=range(4, 17), bidirectional=True)

# --------- Manual DFA alpha2 ---------
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

# --------- PNS and SNS Index ---------
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

    norms = {
        'mean_rr': {'mu': 900.0, 'sd': 100.0},
        'rmssd': {'mu': 42.0, 'sd': 15.0},
        'sd1_pct': {'mu': 5.0, 'sd': 2.0},
        'mean_hr': {'mu': 67.0, 'sd': 10.0},
        'stress': {'mu': 10.0, 'sd': 5.0},
        'sd2_pct': {'mu': 20.0, 'sd': 7.0},
    }

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

# --------- Manual Frequency domain ---------
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

    if _scipy_welch is not None:
        nperseg = min(nperseg, n)
        if nperseg < 16:
            return None, None
        noverlap = min(noverlap, nperseg // 2)
        f, p = _scipy_welch(x, fs=fs, window="hann", nperseg=nperseg,
                            noverlap=noverlap, detrend="constant", scaling="density")
        return f.astype(float), p.astype(float)

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

#</editor-fold>

#<editor-fold desc="Main">

############################################################### MAIN

# --------- Manual/Custom HRV Metrics ---------
def compute_metrics_from_ibi_list(ibi_list):

    rr = clean_rr_list(ibi_list)

    if rr.size < 3:
        return {
            "mean_rr": float("nan"), "rmssd": float("nan"), "sdnn": float("nan"), "nn50": float("nan"), "pnn50": float("nan"), "tinn": float("nan"),
            "stress_index": float("nan"), "pns_index": float("nan"), "sns_index": float("nan"),
            "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan"),
            "sd1": float("nan"), "sd2": float("nan"), "sd2_sd1": float("nan"),
            "ap_en": float("nan"), "samp_en": float("nan"), "dfa_a1": float("nan"), "dfa_a2": float("nan"),
        }

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

    pns_index, sns_index = compute_pns_sns(rr_list=rr)

    ap_en = compute_ap_en(rr, m=2)
    samp_en = compute_samp_en(rr, m=2)

    dfa_a1 = compute_dfa_a1(rr)
    dfa_a2 = compute_dfa_a2(rr)

    return {
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


# --------- Manual HR Metrics ---------
def compute_metrics_from_hr_list(hr_list, rr_list):
    arr = np.array(hr_list)
    if arr.size > 0:
        return { "mean_hr": float(np.mean(arr)) }
    else:
        hr_results = compute_hr_mean_std_lib(rr_list)
        mean_hr = hr_results["mean"]
        return { "mean_hr": mean_hr }


# --------- HRV using PyHRV Lib ---------
def compute_metrics_from_ibi_list_lib(ibi_list, id_list):

    rr = clean_rr_list(ibi_list)

    if rr.size < 3:
        return {
            "mean_rr": float("nan"), "rmssd": float("nan"), "rmssd_chunks": float("nan"), "sdnn": float("nan"), "sdnn_chunks": float("nan"),
            "nn50": float("nan"), "nn50_chunks": float("nan"), "pnn50": float("nan"), "pnn50_chunks": float("nan"),
            "tinn": float("nan"), "tinn_chunks": float("nan"),
            "stress_index": float("nan"), "pns_index": float("nan"), "sns_index": float("nan"),
            "lf": float("nan"), "hf": float("nan"), "lf_hf": float("nan"),
            "sd1": float("nan"), "sd2": float("nan"), "sd2_sd1": float("nan"),
            "ap_en": float("nan"), "samp_en": float("nan"), "dfa_a1": float("nan"), "dfa_a2": float("nan"),
        }

    rr_results = compute_rr_mean_lib(rr)
    mean_rr = rr_results["mean"]

    ibi_df = pd.DataFrame({"id": id_list, "value_ibi": ibi_list})
    chunks = split_into_chunks(ibi_df)

    rmssd = compute_rmssd_lib(rr)
    rmssd_chunks = compute_rmssd_chunks_lib(chunks)

    sdnn = compute_sdnn_lib(rr)
    sdnn_chunks = compute_sdnn_chunks_lib(chunks)

    nn50 = compute_nn50_lib(rr)
    nn50_chunks = compute_nn50_chunks_lib(chunks)

    pnn50 = compute_pnn50_lib(rr)
    pnn50_chunks = compute_pnn50_chunks_lib(chunks)

    tinn = compute_tinn_lib(rr)
    tinn_chunks = compute_tinn_chunks_lib(chunks)

    stress_index = compute_stress_index(rr)                 # custom function (no Stress Index in PyHRV)
    pns_index, sns_index = compute_pns_sns(rr_list=rr)      # custom function (no PNS SNS in PyHRV)

    ftt_results = compute_ftt_lib(rr)
    lf_power = ftt_results["lf"]
    hf_power = ftt_results["hf"]
    lf_hf = ftt_results["ratio"]

    sd_results = compute_sd_lib(rr)
    sd1 = sd_results["sd1"]
    sd2 = sd_results["sd2"]
    sd2_sd1 = sd_results["ratio"]

    ap_en = compute_ap_en(rr)       # custom function (no ApEn in PyHRV)
    samp_en = compute_sampen_lib(rr)

    dfa_results = compute_dfa_lib(rr)
    dfa_a1 = dfa_results["dfa_a1"]
    dfa_a2 = dfa_results["dfa_a2"]

    return {
        "mean_rr": mean_rr,
        "rmssd": rmssd,
        "rmssd_chunks": rmssd_chunks,
        "sdnn": sdnn,
        "sdnn_chunks": sdnn_chunks,
        "stress_index": stress_index,
        "pns_index": pns_index,
        "sns_index": sns_index,
        "nn50": nn50,
        "nn50_chunks": nn50_chunks,
        "pnn50": pnn50,
        "pnn50_chunks": pnn50_chunks,
        "tinn": tinn,
        "tinn_chunks": tinn_chunks,
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

#</editor-fold>
