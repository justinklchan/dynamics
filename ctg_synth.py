#!/usr/bin/env python3
"""Physiology-based synthetic intrapartum CTG generator (oxygen-debt state model).

Each synthetic fetus is a small grey-box model driven by uterine contractions:

    uterine pressure u(t) ──► uteroplacental flow ─┐
    cord compression occ(t) ──────────────────────┼─► O2 delivery q(t) ─► fetal O2 x(t)
                                                   │        (first order, tau_x)
    x(t) < x_crit ─► anaerobic metabolism ─► base deficit B(t) (cleared slowly)
    reduced flow ─► CO2 retention ─► pCO2 P(t);    pH(t) = 7.621 - 0.0183 B - 0.043 P
                                                   (regression fitted on CTU-CHB)
    FHR = baseline + drift + tachycardia(B) - bradycardia(decompensation)
          + variability(sleep state, B) + accelerations
          - chemoreflex decel (delayed fall of x: late decelerations)
          - baroreflex decel (cord occlusion: variable decelerations)
          - vagal decel (head compression: early decelerations)

Observed signals add tocodynamometer scaling, signal loss, maternal heart-rate
capture and quantization. Records are written in the CTU-CHB WFDB format (so
ctg_sysid.py runs on them unchanged) together with the hidden trajectories.

Example:
    python3 ctg_synth.py --n 552 --out synth --compare ctu-chb-intrapartum-cardiotocography-database-1.0.0
"""

import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: ctg_synth.py [--n N] [--out DIR] [--seed S] [--compare DIR] [--no-plot] [--no-show]

  --n N          number of synthetic fetuses (default: 552, the size of CTU-CHB)
  --out DIR      output directory (default: ./synth)
  --seed S       random seed (default: 0)
  --compare DIR  real CTU-CHB directory; plots synthetic vs real feature distributions
  --no-plot      skip all figures
  --no-show      save figures without displaying them

Outputs in DIR:
  <id>.dat/.hea      WFDB records (FHR, UC at 4 Hz) with pH/BDecf/pCO2 in the header
  <id>_latent.npz    hidden 1 Hz trajectories (true IUP, occlusion, O2, BD, pH, components)
  RECORDS            record list
  params.csv         per-fetus parameters and outcomes
  examples.png       example records with hidden states
  compare.png        synthetic vs real feature ECDFs (with --compare)
""")
    sys.exit(0)

import argparse
import csv
import re
from typing import Optional
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=552)
ap.add_argument('--out', default='synth')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--compare', default=None)
ap.add_argument('--no-plot', action='store_true')
ap.add_argument('--no-show', action='store_true')
args = ap.parse_args()
if args.n < 1:
    sys.exit('error: --n must be positive')
if args.compare and not (Path(args.compare) / 'RECORDS').exists():
    sys.exit(f'error: {args.compare}/RECORDS not found')

import numpy as np
from scipy import signal

FS = 4.0                  # CTU-CHB sampling rate (Hz)
WARMUP_MIN = 60           # unrecorded labour simulated before the record starts
PH_COEF = (7.621, -0.0183, -0.0430)   # pH = c0 + c1*BDecf + c2*pCO2 (CTU-CHB fit, resid SD 0.014)
PH_NOISE = 0.014
BD_MAX = 20.0             # reject fetuses beyond survivable acidosis (not seen in CTU-CHB)

# Prior over fetus/labour parameters. Each entry is a callable of (rng, r) where r in [0, 1]
# is a latent "placental compromise" level shared by the placental parameters.
PRIOR = {
    # placenta and oxygenation
    'R':         lambda g, r: 1 - 0.45 * r * g.uniform(0.5, 1.0),     # O2 delivery reserve
    'a_up':      lambda g, r: np.clip(0.15 + 0.60 * r + g.normal(0, 0.05), 0.05, 0.9),
    'tau_x':     lambda g, r: g.uniform(30, 60),       # s, fetal O2 time constant
    'x_crit':    lambda g, r: g.uniform(0.55, 0.65),   # anaerobic threshold
    'k_lac':     lambda g, r: g.uniform(2.5, 4.0),     # mmol/L/min per unit O2 deficit
    'tau_clr':   lambda g, r: g.uniform(180, 300),     # min, base-deficit clearance
    'B_init':    lambda g, r: max(0.0, g.normal(1.6, 1.8) + 3 * r + g.exponential(1.8)),
    'B_push':    lambda g, r: g.uniform(0.5, 1.5),     # mmol/L/h maternal-pushing load, stage 2
    'P0':        lambda g, r: g.normal(6.3, 0.9),      # kPa
    'k_co2':     lambda g, r: g.uniform(3.0, 8.0),     # kPa per unit flow reduction
    # contractions
    'ci1':       lambda g, r: g.uniform(2.3, 5.0),     # min, mean interval, first stage
    'ci2':       lambda g, r: g.uniform(1.5, 2.5),     # min, second stage
    'c_amp':     lambda g, r: g.uniform(35, 60),       # mmHg above tone
    'c_dur':     lambda g, r: g.uniform(55, 90),       # s
    'tone':      lambda g, r: g.uniform(8, 14),        # mmHg
    # cord and head compression
    'p_cord1':   lambda g, r: g.beta(2, 3),
    'p_cord2':   lambda g, r: g.beta(1.5, 3),
    's_cord':    lambda g, r: g.uniform(0.4, 1.0),     # max occlusion fraction
    'p_head2':   lambda g, r: g.beta(2, 3),
    # reflexes and heart rate
    'fhr0':      lambda g, r: g.normal(135, 8),
    'chemo_del': lambda g, r: g.uniform(8, 18),        # s, chemoreceptor + transport delay
    'chemo_thr': lambda g, r: g.uniform(0.04, 0.10),   # O2 fall below adapted level before reflex
    'g_chemo':   lambda g, r: g.uniform(90, 160),      # bpm per unit O2 fall
    'g_baro':    lambda g, r: g.uniform(35, 70),       # bpm at full occlusion
    'g_head':    lambda g, r: g.uniform(8, 20),        # bpm at contraction peak
    'g_os':      lambda g, r: g.uniform(0.2, 0.6),     # post-deceleration overshoot gain
    'B_tach':    lambda g, r: g.uniform(6, 9),
    'g_tach':    lambda g, r: g.uniform(8, 22),
    'B_dec':     lambda g, r: g.uniform(11, 15),
    'g_brady':   lambda g, r: g.uniform(30, 60),
    'sigma0':    lambda g, r: g.lognormal(np.log(5.2), 0.3),  # bpm, variability SD
    'acc_rate':  lambda g, r: g.uniform(0.15, 0.40),   # per min in active sleep
    # recording
    'T_min':     lambda g, r: float(np.clip(g.normal(72, 8), 60, 90)),
    'T2_min':    lambda g, r: 0.0 if g.random() < 0.1 else float(min(30, g.gamma(2, 6))),
    'miss_f':    lambda g, r: g.beta(1.5, 5.0),        # FHR signal-loss fraction
    'miss_u':    lambda g, r: g.beta(1.0, 5.0),        # UC signal-loss fraction
    'toco_gain': lambda g, r: g.lognormal(0, 0.3),
    'toco_off':  lambda g, r: g.uniform(0, 12),
    'p_mhr':     lambda g, r: g.uniform(0, 0.3),       # prob. of maternal HR capture episodes
}


def sample_params(rng: np.random.Generator) -> dict:
    """Draw one fetus's parameters from PRIOR."""
    r = rng.beta(1.2, 2.2)
    p = {k: float(f(rng, r)) for k, f in PRIOR.items()}
    p['r'] = float(r)
    return p


# ---------------------------------------------------------------- primitives
def lowpass1(x: np.ndarray, tau_s: float, y0: Optional[float] = None) -> np.ndarray:
    """First-order lag dy/dt = (x - y)/tau, exact for piecewise-constant input at FS."""
    a = np.exp(-1 / (FS * tau_s))
    y0 = x[0] if y0 is None else y0
    return signal.lfilter([1 - a], [1, -a], x, zi=[a * y0])[0]


def ou(n: int, tau_s: float, sd: float, rng) -> np.ndarray:
    """Ornstein-Uhlenbeck process with stationary SD `sd`."""
    a = np.exp(-1 / (FS * tau_s))
    w = rng.standard_normal(n) * sd * np.sqrt(1 - a * a)
    return signal.lfilter([1], [1, -a], w, zi=[sd * rng.standard_normal() * a])[0]


def delay(x: np.ndarray, d_s: float) -> np.ndarray:
    k = int(round(d_s * FS))
    return np.r_[np.full(k, x[0]), x[:len(x) - k]] if k else x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bump(n: int, centre: float, half_width: float, amp: float, out: np.ndarray):
    """Add a raised-cosine bump (in samples) to `out` in place."""
    a, b = int(max(0, centre - half_width)), int(min(n, centre + half_width))
    if b > a:
        t = (np.arange(a, b) - centre) / half_width
        out[a:b] += amp * 0.5 * (1 + np.cos(np.pi * t))


# ---------------------------------------------------------------- forcing
def contractions(p: dict, n: int, i2: int, rng) -> dict:
    """True intrauterine pressure (above tone), cord occlusion and head compression."""
    u = np.zeros(n)
    occ = np.zeros(n)
    head = np.zeros(n)
    t = rng.uniform(0, 2) * 60 * FS
    while t < n:
        stage2 = t >= i2
        D = p['c_dur'] * rng.uniform(0.8, 1.2) * FS
        A = p['c_amp'] * rng.uniform(0.75, 1.25) * (1.2 if stage2 else 1.0)
        a, b = int(t), int(min(n, t + D))
        ph = (np.arange(a, b) - t) / D
        shape = np.sin(np.pi * ph) ** 2 * (1 + 0.3 * (0.5 - ph))   # slightly faster rise
        shape /= shape.max() if shape.size and shape.max() > 0 else 1
        u[a:b] += A * shape
        if stage2:   # maternal pushing efforts on top of the contraction
            for _ in range(rng.integers(2, 5)):
                bump(n, t + D * rng.uniform(0.25, 0.75), rng.uniform(6, 14) * FS,
                     rng.uniform(20, 50), u)
        if rng.random() < (p['p_cord2'] if stage2 else p['p_cord1']):
            thr = rng.uniform(0.2, 0.5)
            sev = p['s_cord'] * rng.uniform(0.6, 1.0)
            occ[a:b] = np.maximum(occ[a:b], sev * np.clip((shape - thr) / (1 - thr), 0, 1))
        if stage2 and rng.random() < p['p_head2']:
            head[a:b] = np.maximum(head[a:b], shape)
        mean_i = (p['ci2'] if stage2 else p['ci1']) * 60 * FS
        t += max(D + 20 * FS, rng.gamma(11, mean_i / 11))
    return dict(u=u, occ=occ, head=head)


def sleep_states(n: int, rng) -> np.ndarray:
    """Alternating quiet (0) / active (1) behavioural states."""
    s = np.zeros(n)
    state = int(rng.random() < 0.65)
    i = 0
    while i < n:
        L = int((rng.uniform(20, 50) if state else rng.uniform(15, 30)) * 60 * FS)
        L = int(L * rng.uniform(0, 1)) if i == 0 else L        # random phase at start
        s[i:i + L] = state
        i += L
        state = 1 - state
    return s


# ---------------------------------------------------------------- physiology
def simulate(p: dict, rng) -> dict:
    """Simulate one fetus; returns observed FHR/UC (recorded part) and hidden states."""
    nrec = int(p['T_min'] * 60 * FS)
    nw = int(WARMUP_MIN * 60 * FS)
    n = nw + nrec
    i2 = n - int(p['T2_min'] * 60 * FS)               # second stage onset
    f = contractions(p, n, i2, rng)
    u, occ, head = f['u'], f['occ'], f['head']

    # oxygen delivery and fetal oxygenation
    q_flow = (1 - p['a_up'] * np.clip(u / 60, 0, 1.3)) * (1 - occ)
    q_flow = np.clip(q_flow, 0, 1)
    x = lowpass1(p['R'] * q_flow, p['tau_x'], y0=p['R'])

    # metabolic acidosis, CO2 and pH
    prod = p['k_lac'] / 60 * np.maximum(0, p['x_crit'] - x)            # mmol/L/s
    prod[i2:] += p['B_push'] / 3600
    tau_b = p['tau_clr'] * 60
    B = lowpass1(p['B_init'] + tau_b * prod, tau_b, y0=p['B_init'])
    P = lowpass1(p['P0'] + p['k_co2'] * (1 - q_flow), 180.0)
    pH = PH_COEF[0] + PH_COEF[1] * B + PH_COEF[2] * P

    # autonomic / myocardial state
    c = lowpass1(sigmoid((B - p['B_tach']) / 1.5), 600.0, y0=0)   # catecholamine tachycardia
    z = lowpass1(sigmoid((B - p['B_dec']) / 1.0), 300.0, y0=0)    # decompensation

    # reflex decelerations
    x_ref = lowpass1(x, 900.0, y0=p['R'])                          # adapted O2 level
    x_d = lowpass1(delay(x, p['chemo_del']), 10.0)
    d_chemo = p['g_chemo'] * (1 + 0.05 * B) * (1 + z) * np.maximum(0, x_ref - p['chemo_thr'] - x_d)
    d_baro = p['g_baro'] * lowpass1(occ, 3.0, y0=0)
    d_head = p['g_head'] * lowpass1(head, 2.0, y0=0)
    # rebound tachycardia after decelerations, amplified by catecholamines
    d_tot = d_chemo + d_baro
    over = p['g_os'] * (0.5 + 2 * c) * np.maximum(0, lowpass1(d_tot, 40.0, 0) - lowpass1(d_tot, 8.0, 0))

    # baseline, variability, accelerations
    sleep = sleep_states(n, rng)
    base = (p['fhr0'] + ou(n, 600, 2.5, rng) + ou(n, 120, 1.2, rng)
            + p['g_tach'] * c - p['g_brady'] * z ** 2)
    v = ou(n, 5.0, 1.0, rng) + 0.6 * ou(n, 1.0, 1.0, rng)
    v *= p['sigma0'] / 1.17 * (0.5 + 0.5 * lowpass1(sleep, 60.0)) * (1 + 0.4 * c) * (1 - 0.75 * z)
    acc = np.zeros(n)
    ofac = np.clip((lowpass1(x, 300.0, y0=p['R']) - 0.5) / 0.3, 0, 1) * (1 - 0.7 * c)
    t = 0.0
    while True:
        t += rng.exponential(60 * FS / p['acc_rate'])
        if t >= n:
            break
        k = int(t)
        if sleep[k] and rng.random() < ofac[k]:
            bump(n, t, rng.uniform(10, 30) * FS, rng.uniform(15, 30), acc)
    fhr = np.clip(base + v + acc + over - d_chemo - d_baro - d_head, 50, 210)

    # ---- observation model (recorded part only)
    sl = slice(nw, n)
    rec_i2 = i2 - nw
    fo = fhr[sl].copy()
    if rng.random() < p['p_mhr']:                     # maternal heart-rate capture
        for _ in range(rng.integers(1, 4)):
            L = int(rng.uniform(30, 150) * FS)
            a = int(rng.uniform(max(0, rec_i2 - 20 * 60 * FS), nrec - L)) if nrec > L else 0
            fo[a:a + L] = rng.uniform(80, 115) + ou(L, 20, 4, rng) + 0.3 * u[sl][a:a + L]
    fo = np.round(fo * 4) / 4
    poor = contact_episodes(nrec, rng)               # shared poor-contact periods (FHR and UC)
    keep = dropout(nrec, p['miss_f'], rng, poor)
    if rec_i2 < nrec:
        keep[rec_i2:] &= dropout(nrec - rec_i2, min(0.7, 1.6 * p['miss_f']), rng, poor[rec_i2:])
    fo[~keep] = 0
    uo = np.clip(p['toco_gain'] * (u[sl] + p['tone'] - 8) + p['toco_off']
                 + ou(nrec, 5, 1.0, rng), 0.5, 127)
    uo = np.round(uo * 2) / 2
    uo[~dropout(nrec, p['miss_u'], rng, poor)] = 0

    out_pH = float(pH[-1] + rng.normal(0, PH_NOISE))
    ds = slice(nw, n, int(FS))                        # 1 Hz hidden trajectories
    latent = dict(u_true=u[ds], occ=occ[ds], head=head[ds], x=x[ds], BD=B[ds], pCO2=P[ds],
                  pH=pH[ds], catechol=c[ds], decomp=z[ds], sleep=sleep[ds], baseline=base[ds],
                  d_chemo=d_chemo[ds], d_baro=d_baro[ds], d_head=d_head[ds], overshoot=over[ds], acc=acc[ds],
                  fhr_true=fhr[ds])
    return dict(fhr=fo, uc=uo, i2=max(0, rec_i2), latent={k: v.astype(np.float32)
                                                          for k, v in latent.items()},
                pH=out_pH, BDecf=float(B[-1]), pCO2=float(P[-1]),
                min_x=float(x[sl].min()), BD_rise=float(B[-1] - B[nw]))


def contact_episodes(n: int, rng, good_min: float = 10.0, poor_min: float = 2.5) -> np.ndarray:
    """Alternating good/poor sensor-contact periods (exponential durations); True = poor."""
    poor = np.zeros(n, bool)
    i, state = 0, rng.random() < poor_min / (good_min + poor_min)
    while i < n:
        L = int(rng.exponential(poor_min if state else good_min) * 60 * FS) + 1
        poor[i:i + L] = state
        i, state = i + L, not state
    return poor


def dropout(n: int, frac: float, rng, poor: np.ndarray, short_med_s: float = 2.0,
            short_per_min: float = 1.4, long_med_s: float = 45.0, long_sig: float = 1.1,
            p_in_poor: float = 0.9) -> np.ndarray:
    """Boolean keep-mask covering roughly `frac` of the record with gaps.

    Matches the CTU-CHB gap structure: frequent brief dropouts (median ~2 s) plus a few
    long, heavy-tailed losses that carry most of the missing signal. Gaps start inside
    poor-contact periods with probability `p_in_poor`, which leaves long clean stretches.
    """
    keep = np.ones(n, bool)
    if frac <= 0:
        return keep
    idx_poor = np.flatnonzero(poor)

    def start():
        if idx_poor.size and rng.random() < p_in_poor:
            return idx_poor[rng.integers(idx_poor.size)]
        return rng.integers(0, n)
    for _ in range(rng.poisson(short_per_min * n / FS / 60)):
        a = start()
        keep[a:a + int(rng.lognormal(np.log(short_med_s), 0.9) * FS) + 1] = False
    long_frac = max(0.0, frac - (1 - keep.mean()))
    mean_long = long_med_s * np.exp(long_sig ** 2 / 2)
    for _ in range(rng.poisson(long_frac * n / FS / mean_long)):
        a = start()
        keep[a:a + int(rng.lognormal(np.log(long_med_s), long_sig) * FS)] = False
    return keep


# ---------------------------------------------------------------- I/O
def write_wfdb(out: Path, rid: str, sim: dict, p: dict):
    """Write a CTU-CHB-style record: interleaved int16, gain 100, plus annotated header."""
    x = np.column_stack([np.round(sim['fhr'] * 100), np.round(sim['uc'] * 100)]).astype('<i2')
    x.tofile(out / f'{rid}.dat')
    chk = [int(((x[:, j].astype(np.int64).sum() + 32768) % 65536) - 32768) for j in (0, 1)]
    n = len(x)
    lines = [f'{rid} 2 4 {n}',
             f'{rid}.dat 16 100(0)/bpm 12 0 {x[0, 0]} {chk[0]} 0 FHR',
             f'{rid}.dat 16 100/nd 12 0 {x[0, 1]} {chk[1]} 0 UC',
             '', f'#----- Synthetic record {rid} (ctg_synth.py, oxygen-debt model)', '',
             '#-- Outcome measures',
             f'#pH           {sim["pH"]:.2f}',
             f'#BDecf        {sim["BDecf"]:.2f}',
             f'#pCO2         {sim["pCO2"]:.1f}',
             f'#BE           {-sim["BDecf"]:.1f}', '',
             '#-- Delivery descriptors',
             f'#II.stage     {int(round(p["T2_min"])) if p["T2_min"] > 0 else -1}', '',
             '#-- Signal information',
             f'#Pos. II.st.  {sim["i2"] if p["T2_min"] > 0 else -1}', '',
             '#-- Synthetic parameters']
    lines += [f'#{k:<12} {v:.4g}' for k, v in p.items()]
    (out / f'{rid}.hea').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- features (real and synthetic)
def read_record(d: Path, rid: str) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.fromfile(d / f'{rid}.dat', dtype='<i2').reshape(-1, 2) / 100.0
    m = re.search(r'#pH\s+(\S+)', (d / f'{rid}.hea').read_text())
    return x[:, 0], x[:, 1], float(m.group(1)) if m else np.nan


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    dd = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)))


def record_features(fhr: np.ndarray, uc: np.ndarray) -> dict:
    """Simple clinical-style CTG features computed identically on real and synthetic data."""
    valid = (fhr > 50) & (fhr < 210)
    n1 = len(fhr) // 4
    f4 = np.where(valid, fhr, np.nan)[:n1 * 4].reshape(n1, 4)
    f1 = np.nanmean(f4, axis=1) if n1 else np.array([])
    f1[np.isnan(f4).any(axis=1)] = np.nan
    nm = n1 // 60
    mins = np.array([np.nanmedian(f1[i * 60:(i + 1) * 60]) if np.isfinite(f1[i * 60:(i + 1) * 60]).sum() > 20
                     else np.nan for i in range(nm)])
    bl_m = np.array([np.nanmedian(mins[max(0, i - 5):i + 5]) if np.isfinite(mins[max(0, i - 5):i + 5]).any()
                     else np.nan for i in range(nm)])
    bl = np.repeat(bl_m, 60)
    f1 = f1[:len(bl)]
    hours = np.isfinite(f1).sum() / 3600
    dev = f1 - bl
    dec = [(a, b) for a, b in runs(np.nan_to_num(dev, nan=0) < -15) if b - a >= 15]
    acc = [(a, b) for a, b in runs(np.nan_to_num(dev, nan=0) > 15) if b - a >= 15]
    e = f1[:len(f1) // 4 * 4].reshape(-1, 4).mean(axis=1) if len(f1) >= 4 else np.array([np.nan])
    stv = np.nanmean(np.abs(np.diff(e))) if np.isfinite(np.diff(e)).any() else np.nan
    # FIGO baseline variability: per-minute peak-to-trough amplitude of the 1 Hz trace,
    # excluding decelerations/accelerations (>15 bpm from baseline); median over minutes.
    amps = []
    for i in range(len(bl_m)):
        seg, d = f1[i * 60:(i + 1) * 60], dev[i * 60:(i + 1) * 60]
        v = seg[np.isfinite(seg) & (np.abs(np.nan_to_num(d, nan=99)) <= 15)]
        if len(v) >= 40:
            amps.append(np.percentile(v, 97.5) - np.percentile(v, 2.5))
    uv = uc > 0
    u1 = uc[:n1 * 4].reshape(n1, 4).mean(axis=1)
    u1s = np.convolve(u1, np.ones(10) / 10, 'same')
    pk, _ = signal.find_peaks(u1s, prominence=10, distance=90)
    pk = [k for k in pk if uv[k * 4]]
    return {'baseline (bpm)': np.nanmedian(f1) if hours > 0 else np.nan,
            'STV (bpm)': stv,
            'decelerations / h': len(dec) / hours if hours > 0.1 else np.nan,
            'deceleration area (bpm·min / h)':
                sum(-np.nansum(dev[a:b]) for a, b in dec) / 60 / hours if hours > 0.1 else np.nan,
            'accelerations / h': len(acc) / hours if hours > 0.1 else np.nan,
            'contractions / 10 min': len(pk) / max(uv.mean() * len(uc) / FS / 600, 1e-3),
            'FHR signal loss': 1 - valid.mean(),
            'variability (bpm)': float(np.median(amps)) if amps else np.nan}


# ---------------------------------------------------------------- plots
def setup_plot():
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.family'] = 'Helvetica'
    sns.set_palette('deep')
    return plt, sns.color_palette('deep')


def plot_examples(out: Path, sims: list, plt, pal):
    """FHR/UC with hidden O2 and pH for a normal, an intermediate and an acidemic fetus."""
    order = np.argsort([s['pH'] for s in sims])
    pick = [order[-1], order[len(order) // 4], order[0]]
    fig, axs = plt.subplots(4, 3, figsize=(15, 9), sharex='col',
                            gridspec_kw=dict(height_ratios=[1, 2, 1, 1]))
    for j, i in enumerate(pick):
        s, L = sims[i], sims[i]['latent']
        n = len(s['fhr'])
        t4 = (np.arange(n) - n) / FS / 60
        t1 = (np.arange(len(L['x'])) - len(L['x'])) / 60
        fo = np.where(s['fhr'] > 0, s['fhr'], np.nan)
        uo = np.where(s['uc'] > 0, s['uc'], np.nan)
        ax = axs[0, j]
        ax.plot(t4, uo, color=pal[0], ls='-', lw=0.6, label='toco (observed)')
        ax.plot(t1, L['u_true'] + 10, color=pal[1], ls='--', lw=0.8, label='IUP (true)')
        ax.set_title(f'{s["id"]}: pH {s["pH"]:.2f}, BDecf {s["BDecf"]:.1f}')
        ax.set_ylabel('UC (mmHg)')
        ax = axs[1, j]
        ax.plot(t4, fo, color='0.4', ls='-', lw=0.5, label='FHR (observed)')
        ax.plot(t1, L['baseline'], color=pal[2], ls='--', lw=1.0, label='baseline')
        ax.plot(t1, L['baseline'] - L['d_chemo'], color=pal[3], ls='-.', lw=0.8,
                label='baseline - chemoreflex')
        ax.plot(t1, L['baseline'] - L['d_baro'], color=pal[0], ls=':', lw=0.9,
                label='baseline - baroreflex')
        ax.set_ylim(50, 200)
        ax.set_ylabel('FHR (bpm)')
        ax = axs[2, j]
        ax.plot(t1, L['x'], color=pal[0], ls='-', lw=0.8, label='O2 content x')
        ax.plot(t1, L['occ'], color=pal[4], ls=':', lw=0.8, label='cord occlusion')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('fraction')
        ax = axs[3, j]
        ax.plot(t1, L['pH'], color=pal[3], ls='-', lw=1.2, label='pH (hidden)')
        ax.axhline(7.05, color='0.3', ls='--', lw=0.8, label='pH 7.05')
        ax.set_ylabel('pH')
        ax.set_xlabel('minutes before delivery')
    for ax in axs[:, 0]:
        ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'examples.png', dpi=150)


def plot_compare(out: Path, real: list, synth: list, plt, pal):
    """ECDFs of CTG features: real vs synthetic, split by pH < 7.15."""
    keys = list(real[0][1].keys()) + ['pH']
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    groups = [('real, pH ≥ 7.15', real, False, pal[0], '-'),
              ('real, pH < 7.15', real, True, pal[3], '--'),
              ('synthetic, pH ≥ 7.15', synth, False, pal[0], '-.'),
              ('synthetic, pH < 7.15', synth, True, pal[3], ':')]
    print(f'\n{"feature":34s}' + ''.join(f'{g[0]:>22s}' for g in groups))
    for ax, k in zip(axs.flat, keys):
        row = f'{k:34s}'
        for name, data, acid, col, ls in groups:
            v = np.array([ph if k == 'pH' else f[k] for ph, f in data if np.isfinite(ph)
                          and (ph < 7.15) == acid], float)
            v = np.sort(v[np.isfinite(v)])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=col, ls=ls,
                    lw=1.4, label=f'{name} (n={len(v)})')
            row += f'{np.median(v):22.2f}' if len(v) else f'{"-":>22s}'
        print(row)
        ax.set_xlabel(k)
        ax.set_ylabel('ECDF')
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'compare.png', dpi=150)


# ---------------------------------------------------------------- main
def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sims, rows, ids = [], [], []
    for i in range(args.n):
        rid = str(5001 + i)       # numeric, disjoint from CTU-CHB ids (1001-2046)
        while True:
            p = sample_params(rng)
            s = simulate(p, rng)
            if s['BDecf'] <= BD_MAX:
                break
        s['id'] = rid
        write_wfdb(out, rid, s, p)
        np.savez_compressed(out / f'{rid}_latent.npz', **s['latent'])
        rows.append(dict(id=rid, pH=s['pH'], BDecf=s['BDecf'], pCO2=s['pCO2'],
                         min_x=s['min_x'], BD_rise=s['BD_rise'], **p))
        sims.append(s)
        ids.append(rid)
    (out / 'RECORDS').write_text('\n'.join(ids) + '\n')
    with open(out / 'params.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pH = np.array([s['pH'] for s in sims])
    print(f'wrote {args.n} records to {out}/')
    print(f'pH: median {np.median(pH):.3f}, <7.05: {np.mean(pH < 7.05):.1%}, '
          f'<7.15: {np.mean(pH < 7.15):.1%}, 5-95%: {np.percentile(pH, 5):.2f}-{np.percentile(pH, 95):.2f}')

    if args.no_plot:
        return
    plt, pal = setup_plot()
    plot_examples(out, sims, plt, pal)
    if args.compare:
        d = Path(args.compare)
        real = []
        for rid in (d / 'RECORDS').read_text().split():
            f, u, ph = read_record(d, rid)
            real.append((ph, record_features(f, u)))
        print(f'real pH: median {np.nanmedian([r[0] for r in real]):.3f}, '
              f'<7.05: {np.mean([r[0] < 7.05 for r in real]):.1%}, '
              f'<7.15: {np.mean([r[0] < 7.15 for r in real]):.1%}')
        synth = [(s['pH'], record_features(s['fhr'], s['uc'])) for s in sims]
        plot_compare(out, real, synth, plt, pal)
    plt.show()


if __name__ == '__main__':
    main()
