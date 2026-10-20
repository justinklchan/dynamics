#!/usr/bin/env python3
"""Reproduce Warrick et al. (IEEE TBME 2010) CTG systems modeling on CTU-CHB.

Pipeline: preprocessing -> 20-min epochs -> baseline (linear fit), system
identification (UP->FHR impulse response with surrogate validation), HRV
(AR spectrum, LF/MF/HF bands) -> K-S tests -> SVM per-epoch classifiers ->
cumulative-count detectors. Results are written as JSON for the explainer page.
"""

import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: ctg_sysid.py [--data DIR] [--out FILE] [--reps N] [--jobs N]

  --data DIR   CTU-CHB directory (default: ./ctu-chb-intrapartum-cardiotocography-database-1.0.0)
  --out FILE   output JS data file (default: ./results/ctg_data.js)
  --reps N     repetitions of 10-fold cross-validation (default: 5)
  --jobs N     parallel workers (default: all cores)
""")
    sys.exit(0)

import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='ctu-chb-intrapartum-cardiotocography-database-1.0.0')
ap.add_argument('--out', default='results/ctg_data.js')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--jobs', type=int, default=-1)
args = ap.parse_args()
DATA = Path(args.data)
if not (DATA / 'RECORDS').exists():
    sys.exit(f'error: {DATA}/RECORDS not found')

import numpy as np
from scipy import signal, stats
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

FS = 4.0             # CTU-CHB sampling rate (both FHR and UC)
FS_SI = 0.5          # system identification / baseline rate
FS_HRV = 2.0         # HRV rate
GAP_BRIDGE_S = 15    # bridge interruptions shorter than this
EPOCH_S, STEP_S = 1200, 600
M_IRF = 60           # IRF memory: 60 samples at 0.5 Hz = 120 s
D_RANGE = np.arange(-30, 31)   # candidate delays: -60 s .. +60 s
N_SURR = 99
BANDS = {'LF': (0.03, 0.15), 'MF': (0.15, 0.5), 'HF': (0.5, 1.0)}
PSD_F = np.linspace(1 / 120, 1.0, 120)
rng_global = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
def load_record(rid: str) -> dict:
    """Read one CTU-CHB record (16-bit interleaved FHR/UC, gain 100) and header outcomes."""
    x = np.fromfile(DATA / f'{rid}.dat', dtype='<i2').reshape(-1, 2) / 100.0
    hdr = (DATA / f'{rid}.hea').read_text()

    def field(key):
        m = re.search(r'#' + re.escape(key) + r'\s+(\S+)', hdr)
        try:
            return float(m.group(1))
        except (AttributeError, ValueError):
            return float('nan')

    return dict(id=rid, fhr=x[:, 0], uc=x[:, 1], pH=field('pH'), BDecf=field('BDecf'),
                apgar5=field('Apgar5'), stage2=field('II.stage'))


def bridge(x: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate invalid runs no longer than max_gap samples."""
    x = x.copy()
    v = valid.copy()
    idx = np.flatnonzero(~valid)
    if idx.size == 0:
        return x, v
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for r in runs:
        a, b = r[0] - 1, r[-1] + 1
        if len(r) <= max_gap and a >= 0 and b < len(x):
            x[r] = np.interp(r, [a, b], [x[a], x[b]])
            v[r] = True
    return x, v


def clean_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return [start, stop) runs of True at least min_len long."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


# ---------------------------------------------------------------- filters
SOS_LP45 = signal.butter(4, 0.0045, fs=FS_SI, output='sos')
SOS_LP30 = signal.butter(4, 0.030, fs=FS_SI, output='sos')
SOS_LP30_4 = signal.butter(4, 0.030, fs=FS, output='sos')


def zfilt(sos, x):
    return signal.sosfiltfilt(sos, x, padtype='even', padlen=min(len(x) - 1, 600))


# ---------------------------------------------------------------- system ID
def lag_matrix(u: np.ndarray, d: int, M: int, n0: int, n1: int) -> np.ndarray:
    """Rows n in [n0, n1): [u[n-d], u[n-d-1], ..., u[n-d-M+1]]."""
    n = np.arange(n0, n1)[:, None]
    return u[n - d - np.arange(M)[None, :]]


def pca_irf_fit(U: np.ndarray, F: np.ndarray):
    """Pseudo-inverse IRF fit keeping components chosen by MDL.

    U: (N, M) lagged input; F: (N, K) outputs (K signals fit at once).
    Returns g (M, K) regression IRF and VAF (K,) in percent.
    """
    N = U.shape[0]
    lam, V = np.linalg.eigh(U.T @ U / N)
    lam = np.maximum(lam, 1e-12)
    phi = V.T @ (U.T @ F) / N                     # (M, K)
    theta = phi / lam[:, None]
    contrib = lam[:, None] * theta ** 2            # output variance per component
    order = np.argsort(-contrib, axis=0)
    c_sorted = np.take_along_axis(contrib, order, axis=0)
    fvar = (F ** 2).mean(axis=0)
    sse = fvar[None, :] - np.cumsum(c_sorted, axis=0)
    k = np.arange(1, U.shape[1] + 1)[:, None]
    mdl = (1 + k * np.log(N) / N) * np.maximum(sse, 1e-12)
    kbest = np.argmin(mdl, axis=0) + 1
    keep = np.zeros_like(theta, dtype=bool)
    for j in range(F.shape[1]):
        keep[order[:kbest[j], j], j] = True
    g = V @ (theta * keep)
    resid = F - U @ g
    vaf = 100 * (1 - resid.var(axis=0) / F.var(axis=0))
    return g, vaf


def aaft(x: np.ndarray, n: int, rng) -> np.ndarray:
    """Amplitude-adjusted Fourier-transform surrogates (Theiler et al. 1992)."""
    N = len(x)
    xs = np.sort(x)
    out = np.empty((N, n))
    for j in range(n):
        ranks = np.argsort(np.argsort(x))
        g = np.sort(rng.standard_normal(N))[ranks]        # gaussianize
        G = np.fft.rfft(g)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(G)))
        ph[0] = 1
        gs = np.fft.irfft(np.abs(G) * ph * np.exp(1j * np.angle(G[0])), N)
        out[:, j] = xs[np.argsort(np.argsort(gs))]        # restore amplitudes
    return out


def system_id(u: np.ndarray, f: np.ndarray, rng) -> dict:
    """Two-sided IRF identification with delay search and surrogate validation."""
    u = u - u.mean()
    f = f - f.mean()
    N = len(u)
    n0, n1 = D_RANGE.max() + M_IRF - 1, N + D_RANGE.min()
    F = np.column_stack([f, aaft(f, N_SURR, rng)])
    F = F - F.mean(axis=0)
    Ft = F[n0:n1]
    vaf_d = np.empty((len(D_RANGE), F.shape[1]))
    gs = []
    for i, d in enumerate(D_RANGE):
        g, vaf = pca_irf_fit(lag_matrix(u, d, M_IRF, n0, n1), Ft)
        vaf_d[i] = vaf
        gs.append(g[:, 0])
    best = np.argmax(vaf_d, axis=0)
    vaf_best = vaf_d[best, np.arange(F.shape[1])]
    i0 = best[0]
    d = int(D_RANGE[i0])
    g = gs[i0]
    rank = int(1 + np.sum(vaf_best[1:] >= vaf_best[0]))
    pred = np.full(N, np.nan)
    pred[n0:n1] = lag_matrix(u, d, M_IRF, n0, n1) @ g
    return dict(d=d * 2.0, G=float(g.sum()), vaf=float(vaf_best[0]), rank=rank,
                valid=bool(rank <= 5 and vaf_best[0] > 0), g=g, pred=pred,
                surr_vaf=vaf_best[1:], vaf_d=vaf_d[:, 0], u=u, f=f)


# ---------------------------------------------------------------- HRV
def levinson(r: np.ndarray, pmax: int):
    """Levinson-Durbin: AR coefficients and prediction error for orders 1..pmax."""
    a = np.zeros(pmax + 1)
    a[0] = 1.0
    e = r[0]
    coefs, errs = [], []
    for p in range(1, pmax + 1):
        k = -(r[p] + a[1:p] @ r[p - 1:0:-1]) / e
        a[1:p + 1] = a[1:p + 1] + k * np.r_[a[p - 1:0:-1], 1.0]
        e *= (1 - k * k)
        coefs.append(a[1:p + 1].copy())
        errs.append(e)
    return coefs, np.array(errs)


def hrv_model(x: np.ndarray) -> dict:
    """AR model (order by MDL) and band powers of the HRV signal."""
    x = x - x.mean()
    N = len(x)
    r = np.correlate(x, x, 'full')[N - 1:N + 41] / N
    coefs, errs = levinson(r, 40)
    p = np.arange(1, 41)
    mdl = N * np.log(errs) + p * np.log(N)
    pb = int(np.argmin(mdl))
    a, s2 = coefs[pb], errs[pb]
    z = np.exp(-2j * np.pi * PSD_F[:, None] / FS_HRV * np.arange(1, len(a) + 1)[None, :])
    psd = s2 / np.abs(1 + z @ a) ** 2
    rel = psd / psd.sum()
    var = x.var()
    out = {f'HRV_{b}': float(np.sqrt(rel[(PSD_F > lo) & (PSD_F <= hi)].sum() * var))
           for b, (lo, hi) in BANDS.items()}
    out['p'] = pb + 1
    out['psd'] = psd * var / psd.sum()
    pred = signal.lfilter(np.r_[0, -a], 1, x)
    out['vaf'] = float(100 * (1 - (x - pred)[50:].var() / x[50:].var()))
    return out


# ---------------------------------------------------------------- per record
def process_record(rid: str, keep_detail: bool) -> dict:
    rec = load_record(rid)
    rng = np.random.default_rng(int(rid))
    fhr, uc = rec['fhr'], rec['uc']
    gap = int(GAP_BRIDGE_S * FS)
    f_b, fv = bridge(fhr, (fhr >= 50) & (fhr <= 210), gap)
    u_b, uv = bridge(uc, uc > 0, gap)
    mask = fv & uv
    T = len(fhr) / FS
    epochs = []
    for a, b in clean_segments(mask, int(EPOCH_S * FS)):
        fs4, us4 = f_b[a:b], u_b[a:b]
        f05 = signal.decimate(signal.decimate(fs4, 2, ftype='fir', zero_phase=True), 4,
                              ftype='fir', zero_phase=True)
        u05 = signal.decimate(signal.decimate(us4, 2, ftype='fir', zero_phase=True), 4,
                              ftype='fir', zero_phase=True)
        f_bl = zfilt(SOS_LP45, f05)
        f_si = zfilt(SOS_LP30, f05) - f_bl
        u_si = zfilt(SOS_LP30, u05) - zfilt(SOS_LP45, u05)
        hrv4 = fs4 - zfilt(SOS_LP30_4, fs4)
        hrv2 = signal.decimate(hrv4, 2, ftype='fir', zero_phase=True)
        seg_s = (b - a) / FS
        starts = list(np.arange(0, seg_s - EPOCH_S + 1, STEP_S))
        if seg_s - (starts[-1] + EPOCH_S) >= 1:
            starts.append(seg_s - EPOCH_S)
        for s0 in starts:
            i05 = slice(int(s0 * FS_SI), int(s0 * FS_SI) + int(EPOCH_S * FS_SI))
            i2 = slice(int(s0 * FS_HRV), int(s0 * FS_HRV) + int(EPOCH_S * FS_HRV))
            t_start = a / FS + s0
            t_end = t_start + EPOCH_S
            bl = f_bl[i05]
            tt = np.arange(len(bl)) / FS_SI / 60
            slope, icpt = np.polyfit(tt, bl, 1)
            si = system_id(u_si[i05], f_si[i05], rng)
            hv = hrv_model(hrv2[i2])
            ep = dict(t0=t_start, t1=t_end, k=int(round((t_end - T) / STEP_S)),
                      d=si['d'], G=si['G'], vaf=si['vaf'], rank=si['rank'], valid=si['valid'],
                      FHR_BL=float(icpt + slope * tt.mean()), BL_slope=float(slope),
                      HRV_LF=hv['HRV_LF'], HRV_MF=hv['HRV_MF'], HRV_HF=hv['HRV_HF'],
                      ar_p=hv['p'], hrv_vaf=hv['vaf'])
            if keep_detail:
                r2 = lambda v: np.round(v, 2).tolist()
                ep['detail'] = dict(
                    u=r2(si['u']), f=r2(si['f']),
                    pred=[None if np.isnan(v) else round(float(v), 2) for v in si['pred']],
                    g=np.round(si['g'], 4).tolist(), surr=r2(si['surr_vaf']),
                    vaf_d=r2(si['vaf_d']), bl=r2(bl), bl_fit=[float(icpt), float(slope)],
                    hrv=r2(hrv2[i2]), psd=np.round(hv['psd'], 4).tolist())
            epochs.append(ep)
    out = dict(id=rid, pH=rec['pH'], BDecf=rec['BDecf'], apgar5=rec['apgar5'],
               dur=T, clean=float(mask.mean()), epochs=epochs)
    if keep_detail:
        raw_f = np.where(fhr > 0, fhr, np.nan)[::4]
        raw_u = np.where(uc > 0, uc, np.nan)[::4]
        nn = lambda v: [None if np.isnan(x) else round(float(x), 1) for x in v]
        out['raw'] = dict(fhr=nn(raw_f), uc=nn(raw_u))
    return out


def label(r: dict):
    """Pathological: pH < 7.05. Normal: pH >= 7.15 and BDecf < 8. Else intermediate."""
    if r['pH'] < 7.05:
        return 1
    if r['pH'] >= 7.15 and not (r['BDecf'] >= 8):
        return 0
    return None


# ---------------------------------------------------------------- classification
FEATS = {'S': ['d', 'G'], 'B': ['FHR_BL', 'HRV_LF', 'HRV_MF']}
GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0]
ALPHAS = [1, 2, 4, 8, 16, 32, 64]
THRESH = [1, 2, 3]
MIN_VALID = 1        # paper: 3 valid epochs in 3-h records; CTU-CHB records are 60-90 min


def pauc(fpr, tpr, fmax=0.3):
    """Area under ROC for FPR in [0, fmax] (unnormalized, max = fmax)."""
    pts = sorted(set(zip(np.r_[0, fpr, 1], np.r_[0, tpr, 1])))
    xs, ys = np.array(pts).T
    ys = np.maximum.accumulate(ys)
    grid = np.linspace(0, fmax, 61)
    return float(np.trapz(np.interp(grid, xs, ys), grid))


def fit_predict(X, y, Xt, gamma, alpha):
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0, class_weight={0: 1, 1: alpha})
    clf.fit(X, y)
    return clf.predict(Xt)


def detect_rates(pred, fetus, y, k, thr):
    """Per-fetus detection: first epoch index at which cumulative positives >= thr."""
    det = {}
    for fid in np.unique(fetus):
        m = fetus == fid
        order = np.argsort(k[m])
        c = np.cumsum(pred[m][order])
        hit = np.flatnonzero(c >= thr)
        det[fid] = (k[m][order][hit[0]] if hit.size else np.inf, y[m][0])
    return det


def det_roc(pred, fetus, y, k):
    tpr, fpr = [], []
    for thr in THRESH:
        det = detect_rates(pred, fetus, y, k, thr)
        lab = np.array([v[1] for v in det.values()])
        hit = np.array([np.isfinite(v[0]) for v in det.values()])
        tpr.append(hit[lab == 1].mean())
        fpr.append(hit[lab == 0].mean())
    return np.array(fpr), np.array(tpr)


def run_fold(Xall, y, fetus, k, tr, te, seed):
    """Nested selection of gamma (per-epoch pAUC) then alpha (detector pAUC)."""
    res = {}
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    itr, iva = next(inner.split(tr, y[tr], fetus[tr]))
    itr, iva = tr[itr], tr[iva]
    for name, X in Xall.items():
        sc = StandardScaler().fit(X[itr])
        Xa, Xv = sc.transform(X[itr]), sc.transform(X[iva])
        best_g, best_a = None, -1
        preds_v = {}
        for gm in GAMMAS:
            fpr, tpr = [], []
            for al in ALPHAS:
                p = fit_predict(Xa, y[itr], Xv, gm, al)
                preds_v[gm, al] = p
                tpr.append(p[y[iva] == 1].mean())
                fpr.append(p[y[iva] == 0].mean())
            a = pauc(np.array(fpr), np.array(tpr))
            if a > best_a:
                best_g, best_a = gm, a
        best_al, best_d = ALPHAS[0], -1
        for al in ALPHAS:
            fpr, tpr = det_roc(preds_v[best_g, al], fetus[iva], y[iva], k[iva])
            a = pauc(fpr, tpr)
            if a > best_d:
                best_al, best_d = al, a
        sc = StandardScaler().fit(X[tr])
        Xa, Xt = sc.transform(X[tr]), sc.transform(X[te])
        roc = []
        for al in ALPHAS:
            p = fit_predict(Xa, y[tr], Xt, best_g, al)
            roc.append(p)
            if al == best_al:
                res[name] = p
        res[name + '_roc'] = roc
        res[name + '_hp'] = (best_g, best_al)
    return te, res


def clean_nan(o):
    """Recursively replace NaN/inf floats with None for strict JSON."""
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    rids = (DATA / 'RECORDS').read_text().split()
    meta = {r: load_record(r) for r in rids}
    labels = {r: label(m) for r, m in meta.items()}
    print(f'{len(rids)} records; normal={sum(v == 0 for v in labels.values())} '
          f'pathological={sum(v == 1 for v in labels.values())}', flush=True)

    recs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process_record)(r, False) for r in rids)
    by_id = {r['id']: r for r in recs}
    for r in recs:
        r['label'] = labels[r['id']]
        r['n_valid'] = sum(e['valid'] for e in r['epochs'])

    included = [r for r in recs if r['label'] is not None and r['n_valid'] >= MIN_VALID]
    n_inc = {c: sum(r['label'] == c for r in included) for c in (0, 1)}
    print(f'included: normal={n_inc[0]} pathological={n_inc[1]}', flush=True)

    # Explorer records: the cleanest normal and pathological cases
    pick = []
    for c in (0, 1):
        cand = sorted([r for r in included if r['label'] == c],
                      key=lambda r: (-r['n_valid'], -r['clean']))
        pick += [r['id'] for r in cand[:8]]
    detail = Parallel(n_jobs=args.jobs)(delayed(process_record)(r, True) for r in pick)
    for d in detail:
        d['label'] = labels[d['id']]

    # Instances: valid-SI epochs of included fetuses
    rows = [(r['id'], r['label'], e) for r in included for e in r['epochs'] if e['valid']]
    fetus = np.array([int(a) for a, _, _ in rows])
    y = np.array([b for _, b, _ in rows])
    k = np.array([e['k'] for _, _, e in rows])
    Xall = {n: np.array([[e[f] for f in fs] for _, _, e in rows]) for n, fs in FEATS.items()}
    print(f'instances: {len(y)} ({y.sum()} pathological)', flush=True)

    # Time progression + K-S per epoch index
    feat_names = ['d', 'G', 'FHR_BL', 'HRV_LF', 'HRV_MF', 'HRV_HF', 'vaf']
    ks_rows = []
    for kk in sorted(set(k)):
        m = k == kk
        row = {'k': int(kk)}
        for fn in feat_names:
            v = np.array([e[fn] for _, _, e in rows])[m]
            yy = y[m]
            a, b = v[yy == 0], v[yy == 1]
            p = stats.ks_2samp(a, b).pvalue if len(a) > 2 and len(b) > 2 else None
            row[fn] = dict(m0=float(a.mean()) if len(a) else None,
                           s0=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
                           m1=float(b.mean()) if len(b) else None,
                           s1=float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else None,
                           n0=int(len(a)), n1=int(len(b)), p=p)
        ks_rows.append(row)
    ks_all = {}
    for fn in feat_names:
        v = np.array([e[fn] for _, _, e in rows])
        ks_all[fn] = float(stats.ks_2samp(v[y == 0], v[y == 1]).pvalue)

    # Cross-validated classification + detection
    jobs = []
    for rep in range(args.reps):
        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rep)
        for tr, te in cv.split(y, y, fetus):
            jobs.append((rep, tr, te))
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_fold)(Xall, y, fetus, k, tr, te, 100 * rep + i)
        for i, (rep, tr, te) in enumerate(jobs))

    kmin = int(k.min())
    ks_axis = list(range(kmin, 1))
    curves = {n: {'sens': np.zeros(len(ks_axis)), 'spec': np.zeros(len(ks_axis))}
              for n in ['S1', 'S2', 'B1', 'B2', 'C1', 'C2']}
    roc = {'S': [], 'B': []}
    hps = {'S': [], 'B': []}
    for rep in range(args.reps):
        pred = {n: np.zeros(len(y), dtype=int) for n in ('S', 'B')}
        for (r, _, _), (te, res) in zip(jobs, out):
            if r != rep:
                continue
            for n in ('S', 'B'):
                pred[n][te] = res[n]
        pred['C'] = pred['S'] | pred['B']
        for n in ('S', 'B', 'C'):
            for thr in (1, 2):
                det = detect_rates(pred[n], fetus, y, k, thr)
                tdet = np.array([v[0] for v in det.values()])
                lab = np.array([v[1] for v in det.values()])
                for i, kk in enumerate(ks_axis):
                    curves[f'{n}{thr}']['sens'][i] += np.mean(tdet[lab == 1] <= kk) / args.reps
                    curves[f'{n}{thr}']['spec'][i] += np.mean(~(tdet[lab == 0] <= kk)) / args.reps
    # Per-epoch ROC from pooled out-of-fold predictions, one curve per repetition
    for (_, _, _), (_, res) in zip(jobs, out):
        for n in ('S', 'B'):
            hps[n].append(res[n + '_hp'])
    roc_out = {}
    for n in ('S', 'B'):
        per_rep = []
        for rep in range(args.reps):
            P = np.zeros((len(ALPHAS), len(y)), dtype=int)
            for (r, _, _), (te, res) in zip(jobs, out):
                if r == rep:
                    P[:, te] = np.array(res[n + '_roc'])
            per_rep.append([(P[i][y == 0].mean(), P[i][y == 1].mean()) for i in range(len(ALPHAS))])
        a = np.array(per_rep)                      # reps x alphas x (fpr, tpr)
        pa = [pauc(f[:, 0], f[:, 1]) for f in a]
        se = np.sqrt(a[:, :, 1].mean(0) * (1 - a[:, :, 1].mean(0)) / max(1, int(y.sum())))
        roc_out[n] = dict(fpr=a[:, :, 0].mean(0).tolist(), tpr=a[:, :, 1].mean(0).tolist(),
                          tpr_se=se.tolist(), pauc=float(np.mean(pa)),
                          pauc_se=float(np.std(pa) / np.sqrt(len(pa))))

    # Decision surface (Fig. 5): SI classifier on 90% of fetuses, most common hyperparameters
    g_mode = max(set(hps['S']), key=hps['S'].count)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    tr, te = next(cv.split(y, y, fetus))
    sc = StandardScaler().fit(Xall['S'][tr])
    clf = SVC(kernel='rbf', gamma=g_mode[0], C=1.0, class_weight={0: 1, 1: g_mode[1]})
    clf.fit(sc.transform(Xall['S'][tr]), y[tr])
    dg = np.linspace(-60, 60, 61)
    gl, gh = np.percentile(Xall['S'][:, 1], [1, 99])
    gg = np.linspace(gl, gh, 61)
    DD, GG = np.meshgrid(dg, gg)
    H = clf.decision_function(sc.transform(np.column_stack([DD.ravel(), GG.ravel()])))
    test_path = [f for f in np.unique(fetus[te]) if y[fetus == f][0] == 1]
    traj = []
    for f in test_path:
        m = fetus == f
        o = np.argsort(k[m])
        traj.append(dict(id=str(f), d=Xall['S'][m][o, 0].tolist(), G=Xall['S'][m][o, 1].tolist(),
                         H=clf.decision_function(sc.transform(Xall['S'][m][o])).tolist()))
    surface = dict(d=dg.tolist(), G=gg.tolist(), H=np.round(H.reshape(GG.shape), 3).tolist(),
                   pts=dict(d=Xall['S'][tr, 0].tolist(), G=Xall['S'][tr, 1].tolist(),
                            y=y[tr].tolist(), sv=np.isin(np.arange(len(tr)), clf.support_).tolist()),
                   n_sv=int(len(clf.support_)), n_train=int(len(tr)),
                   gamma=g_mode[0], alpha=g_mode[1], traj=traj)

    summary_recs = [dict(id=r['id'], label=r['label'], pH=r['pH'], BDecf=r['BDecf'],
                         n_ep=len(r['epochs']), n_valid=r['n_valid'], clean=round(r['clean'], 3))
                    for r in recs]
    payload = dict(
        meta=dict(n_records=len(rids), n_normal=sum(v == 0 for v in labels.values()),
                  n_path=sum(v == 1 for v in labels.values()), included=n_inc,
                  n_instances=int(len(y)), n_inst_path=int(y.sum()), reps=args.reps,
                  n_epochs=int(sum(len(r['epochs']) for r in recs)),
                  n_valid=int(sum(r['n_valid'] for r in recs)),
                  M=M_IRF, fs_si=FS_SI, fs_hrv=FS_HRV, psd_f=PSD_F.round(4).tolist(),
                  d_range=(D_RANGE * 2).tolist()),
        records=summary_recs, detail=detail, ks=ks_rows, ks_all=ks_all,
        roc=roc_out, curves={n: {kk: np.round(v, 4).tolist() for kk, v in c.items()}
                             for n, c in curves.items()},
        k_axis=ks_axis, surface=surface,
        hp=dict(S=[list(h) for h in hps['S']], B=[list(h) for h in hps['B']]))
    payload = clean_nan(payload)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('window.CTG = ' + json.dumps(payload, separators=(',', ':'),
                                                 allow_nan=False, default=float) + ';\n')
    print(f'wrote {outp} ({outp.stat().st_size / 1e6:.1f} MB)')
    print('per-epoch pAUC  S=%.3f  B=%.3f' % (roc_out['S']['pauc'], roc_out['B']['pauc']))
    for n, c in curves.items():
        i = ks_axis.index(-3) if -3 in ks_axis else 0
        print(f'{n}: sens@k=-3 {c["sens"][i]:.2f} spec {c["spec"][i]:.2f} | '
              f'final sens {c["sens"][-1]:.2f} spec {c["spec"][-1]:.2f}')


if __name__ == '__main__':
    main()
