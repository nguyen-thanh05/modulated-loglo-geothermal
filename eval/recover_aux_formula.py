"""Recover closed-form BHP / energy maps from dataset_v3_hetero well columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WELL_COORDS = [
    (31, 15), (45, 4), (56, 15), (45, 27), (18, 27),
    (4, 15), (18, 4), (18, 15), (45, 15),
]
WELL_NAMES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "Inj1", "Inj2"]
N_PROD = 7
PERF = slice(2, 12)
DZ_M = 10.0
DX_M = 10.0
G = 9.80665
RHO_SC = 1000.0
CP_WATER = 4184.0
# SI: 2π k[mD] h[m] ΔP[kPa] / (μ[cp] ln) → m3/day
MD_TO_M2 = 9.869233e-16
WI_SI = 2.0 * np.pi * MD_TO_M2 * 1000.0 / 1e-3 * 86400.0  # 5.3587e-4
RE_PEACEMAN = 0.14 * np.sqrt(DX_M ** 2 + DX_M ** 2)


def load_well_columns_range(path: Path, name: str, start: int, stop: int) -> np.ndarray:
    arr = np.load(path / name, mmap_mode="r")
    cols = np.empty((stop - start, arr.shape[1], arr.shape[2], 9), dtype=np.float32)
    for i, (h, w) in enumerate(WELL_COORDS):
        cols[:, :, :, i] = arr[start:stop, :, :, h, w]
    return cols


def load_well_static_range(path: Path, name: str, start: int, stop: int) -> np.ndarray:
    arr = np.load(path / name, mmap_mode="r")
    cols = np.empty((stop - start, arr.shape[1], 9), dtype=np.float32)
    for i, (h, w) in enumerate(WELL_COORDS):
        cols[:, :, i] = arr[start:stop, :, h, w]
    return cols


def iapws97_region1_enthalpy_jkg(T_C, P_kPa):
    T = np.asarray(T_C, dtype=np.float64) + 273.15
    p_MPa = np.asarray(P_kPa, dtype=np.float64) / 1000.0
    I = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32],
        dtype=np.int32,
    )
    J = np.array(
        [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41],
        dtype=np.int32,
    )
    N = np.array([
        0.14632971213167, -0.84548187169114, -3.7563603672040, 3.3855169168385,
        -0.95791963391359, 0.15772038513228, -0.016616417199501, 8.1214629983568e-4,
        2.8319080123804e-4, -6.0706301565874e-4, -0.018990068218419, -0.032529748770505,
        -0.021841717175414, -5.2838357969930e-5, -4.7184321073267e-4, -3.0001780793026e-4,
        4.7661393906987e-5, -4.4141845330846e-6, -7.2694996297594e-16, -3.1679647391257e-4,
        -2.8270797985312e-6, -8.5205128120103e-10, -2.2425288558572e-3, -6.5171222895601e-7,
        -1.4341729937924e-13, -4.0516996860147e-5, -1.2734301741641e-9, -1.7424871230634e-10,
        -6.8762131295531e-19, 1.4478307828521e-20, 2.6335781662795e-23, -1.1947622640071e-23,
        1.8228094581404e-24, -9.3537087292458e-26,
    ])
    pi = p_MPa / 16.53
    tau = 1386.0 / T
    R = 0.461526
    gamma_tau = np.zeros_like(T, dtype=np.float64)
    for i, j, ni in zip(I, J, N):
        gamma_tau += ni * (pi ** int(i)) * int(j) * (tau ** int(j - 1))
    return tau * R * T * gamma_tau * 1000.0


def mu_cp(T_C):
    T = np.asarray(T_C, dtype=np.float64) + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (T - 140.0)) * 1000.0


def report(name, pred, tgt, mask=None):
    if mask is not None:
        pred = pred[mask]
        tgt = tgt[mask]
    err = pred - tgt
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    maxae = np.max(np.abs(err))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(tgt), 1e-12)) * 100.0
    rel = np.linalg.norm(err) / (np.linalg.norm(tgt) + 1e-12)
    corr = np.corrcoef(pred.ravel(), tgt.ravel())[0, 1] if tgt.size > 1 else np.nan
    print(
        f"  {name:52s}  MAE={mae:.6g}  RMSE={rmse:.6g}  "
        f"max={maxae:.6g}  MAPE={mape:.4g}%  relL2={rel:.4g}  corr={corr:.6f}"
    )
    return dict(mae=mae, rmse=rmse, maxae=maxae, mape=mape, rel=rel, corr=corr)


def k_weighted(field, k):
    w = np.maximum(k, 1e-12)
    return (field * w).sum(axis=-2) / w.sum(axis=-2)


def hydrostatic(P, rho, z_datum, z_index):
    dz = z_datum - z_index.reshape((1,) * (P.ndim - 2) + (-1, 1))
    return P + rho * G * DZ_M * dz / 1000.0


def layer_wi(k_mD, T_C, rw, skin=0.0):
    ln = np.log(RE_PEACEMAN / rw) + skin
    return WI_SI * k_mD * DZ_M / (mu_cp(T_C) * ln)


def peaceman_solve(P, T, k, rho, q, rw, z_datum, producer_mask, skin=0.0):
    """P,T,k,rho: (N,t,Z,W) on perforations. q: (N,t,W) >=0. Returns Pbhp, q_layer."""
    z_index = np.arange(2, 12)
    Ph = hydrostatic(P, rho, z_datum, z_index)
    wi = layer_wi(k, T, rw, skin)
    wi_sum = wi.sum(axis=2)
    Pavg = (wi * Ph).sum(axis=2) / np.maximum(wi_sum, 1e-12)
    # producers: sum wi (Ph - Pbhp) = q  => Pbhp = Pavg - q/wi
    # injectors: sum wi (Pbhp - Ph) = q  => Pbhp = Pavg + q/wi
    sign = np.where(producer_mask, 1.0, -1.0)
    Pbhp = Pavg - sign * q / np.maximum(wi_sum, 1e-12)
    q_layer = wi * (Ph - Pbhp[..., None, :])
    # For injectors q_layer is negative if sign convention is reservoir source.
    return Pbhp, q_layer, wi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset_v3_hetero",
    )
    parser.add_argument("--n-fit", type=int, default=80)
    parser.add_argument("--test-start", type=int, default=350)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()
    data = args.data
    n_fit = args.n_fit
    n_test = args.n_test
    test_start = args.test_start

    print("IAPWS check (expect ~115331 J/kg at 26.85C, 3000 kPa):",
          float(iapws97_region1_enthalpy_jkg(26.85, 3000.0)))
    print("IAPWS 25C, 101.325 kPa:", float(iapws97_region1_enthalpy_jkg(25.0, 101.325)))
    print("IAPWS 170C, 25000 kPa:", float(iapws97_region1_enthalpy_jkg(170.0, 25000.0)))
    print(f"WI_SI={WI_SI:.6e}  re={RE_PEACEMAN:.4f} m")

    print(f"\nLoading well columns from {data} ...")

    def stack_dyn(name):
        a = load_well_columns_range(data, name, 0, n_fit)
        b = load_well_columns_range(data, name, test_start, test_start + n_test)
        return np.concatenate([a, b], axis=0)

    def stack_static(name):
        a = load_well_static_range(data, name, 0, n_fit)
        b = load_well_static_range(data, name, test_start, test_start + n_test)
        return np.concatenate([a, b], axis=0)

    aux_a = np.load(data / "all_energyrate_bhp.npy", mmap_mode="r")
    q_a = np.load(data / "all_action_raw.npy", mmap_mode="r")
    aux = np.concatenate([np.asarray(aux_a[:n_fit]), np.asarray(aux_a[test_start:test_start + n_test])])
    q_raw = np.concatenate([np.asarray(q_a[:n_fit]), np.asarray(q_a[test_start:test_start + n_test])])
    P_f = stack_dyn("all_pres_frac.npy")
    P_m = stack_dyn("all_pres_formation.npy")
    T_f = stack_dyn("all_temp_frac.npy")
    T_m = stack_dyn("all_temp_formation.npy")
    rho_f = stack_dyn("all_density_frac.npy")
    rho_m = stack_dyn("all_density_formation.npy")
    k_f = stack_static("all_perm_frac.npy")
    k_m = stack_static("all_perm_matrix.npy")
    print("Loaded", P_f.shape)

    bhp_all = aux[:, :, :9].astype(np.float64)
    energy_all = aux[:, :, 9:].astype(np.float64)
    q_all = q_raw.astype(np.float64)

    def tp1(field):
        x = field[:, 1:].astype(np.float64)
        return x[:n_fit], x[n_fit:]

    Pf_f, Pf_t = tp1(P_f)
    Pm_f, Pm_t = tp1(P_m)
    Tf_f, Tf_t = tp1(T_f)
    Tm_f, Tm_t = tp1(T_m)
    rf_f, rf_t = tp1(rho_f)
    rm_f, rm_t = tp1(rho_m)
    bhp_f, bhp_t = tp1(bhp_all)
    e_f, e_t = tp1(energy_all)
    q_f, q_t = q_all[:n_fit], q_all[n_fit:]

    def expand_k(k):
        return np.broadcast_to(k[:, None], (k.shape[0], 156, 16, 9)).astype(np.float64)

    kf_f, kf_t = expand_k(k_f[:n_fit]), expand_k(k_f[n_fit:])
    km_f, km_t = expand_k(k_m[:n_fit]), expand_k(k_m[n_fit:])

    print("\n=== Density vs physics.py exponential EOS ===")
    beta, cf = 8.8e-4, 7e-7
    rho_eos = 1000.11 * np.exp(cf * (Pf_t - 100.0) - beta * (Tf_t - 25.0))
    report("rho_frac vs exp EOS (all layers)", rho_eos, rf_t)
    report("rho_frac vs exp EOS (perfs)", rho_eos[:, :, PERF], rf_t[:, :, PERF])

    print("\n=== Mean rates (test) ===")
    print("  producer mean q", q_t[:, :, :N_PROD].mean(), "injector mean q", q_t[:, :, N_PROD:].mean())
    print("  frac dP/dz kPa/layer", np.mean(Pf_t[:, :, 1:, :] - Pf_t[:, :, :-1, :]))
    print("  rho g dz", np.mean(rf_t) * G * DZ_M / 1000.0)

    print("\n=== BHP: simple readouts on test ===")
    report("P_frac[z=2] top perf", Pf_t[:, :, 2, :], bhp_t)
    report("P_frac k-w perfs", k_weighted(Pf_t[:, :, PERF], kf_t[:, :, PERF]), bhp_t)
    Ph1 = hydrostatic(Pf_t[:, :, PERF], rf_t[:, :, PERF], 1, np.arange(2, 12))
    report("hyd z=1 k-w frac", k_weighted(Ph1, kf_t[:, :, PERF]), bhp_t)

    print("\n=== BHP residual structure for hyd z=1 k-w (test) ===")
    pred0 = k_weighted(Ph1, kf_t[:, :, PERF])
    res = bhp_t - pred0
    for w, name in enumerate(WELL_NAMES):
        r = res[:, :, w]
        qq = q_t[:, :, w]
        open_ = qq > 1
        shut = qq < 1e-6
        print(
            f"  {name:5s}  MAE={np.mean(np.abs(r)):7.2f}  "
            f"mean={r.mean():7.2f}  open={r[open_].mean() if open_.any() else np.nan:7.2f}  "
            f"shut={r[shut].mean() if shut.any() else np.nan:7.2f}  "
            f"corr(res,q)={np.corrcoef(r.ravel(), qq.ravel())[0,1]:6.3f}"
        )

    prod_mask = np.array([True] * N_PROD + [False] * 2)

    print("\n=== Metric Peaceman grid-search rw, datum, fracture-only (FIT RMSE, TEST report best) ===")
    best = None
    for z_datum in [0, 1, 2, 3, 5]:
        for rw in [0.05, 0.0762, 0.0889, 0.1, 0.12, 0.15]:
            Pbhp, _, _ = peaceman_solve(
                Pf_f[:, :, PERF], Tf_f[:, :, PERF], kf_f[:, :, PERF],
                rf_f[:, :, PERF], q_f, rw, z_datum, prod_mask,
            )
            rmse = np.sqrt(np.mean((Pbhp - bhp_f) ** 2))
            if best is None or rmse < best[0]:
                best = (rmse, z_datum, rw)
            print(f"  datum={z_datum} rw={rw:.4f}  fit RMSE={rmse:.3f}")
    print("best fracture-only", best)

    z_datum, rw = best[1], best[2]
    Pbhp_t, q_layer_t, wi_t = peaceman_solve(
        Pf_t[:, :, PERF], Tf_t[:, :, PERF], kf_t[:, :, PERF],
        rf_t[:, :, PERF], q_t, rw, z_datum, prod_mask,
    )
    print(f"\nTEST Peaceman frac-only datum={z_datum} rw={rw}")
    report("all wells", Pbhp_t, bhp_t)
    report("producers", Pbhp_t[:, :, :N_PROD], bhp_t[:, :, :N_PROD])
    report("injectors", Pbhp_t[:, :, N_PROD:], bhp_t[:, :, N_PROD:])

    print("\n=== Dual-perm WI = WI_f + WI_m, same rw/datum ===")
    # Parallel: treat as 20 connections by summing WI and WI-weighted P
    def dual_peaceman(Pf, Tf, kf, rf, Pm, Tm, km, rm, q, rw, z_datum):
        z_index = np.arange(2, 12)
        Phf = hydrostatic(Pf, rf, z_datum, z_index)
        Phm = hydrostatic(Pm, rm, z_datum, z_index)
        wif = layer_wi(kf, Tf, rw)
        wim = layer_wi(km, Tm, rw)
        wi = wif + wim
        Pavg = (wif * Phf + wim * Phm).sum(axis=2) / np.maximum(wi.sum(axis=2), 1e-12)
        sign = np.where(prod_mask, 1.0, -1.0)
        Pbhp = Pavg - sign * q / np.maximum(wi.sum(axis=2), 1e-12)
        qf = wif * (Phf - Pbhp[..., None, :])
        qm = wim * (Phm - Pbhp[..., None, :])
        return Pbhp, qf, qm

    Pbhp_d, qf_t, qm_t = dual_peaceman(
        Pf_t[:, :, PERF], Tf_t[:, :, PERF], kf_t[:, :, PERF], rf_t[:, :, PERF],
        Pm_t[:, :, PERF], Tm_t[:, :, PERF], km_t[:, :, PERF], rm_t[:, :, PERF],
        q_t, rw, z_datum,
    )
    report("dual-perm Peaceman", Pbhp_d, bhp_t)

    print("\n=== Fit a single WI prefactor on top of theoretical WI (frac, fit set) ===")
    Pbhp0, _, wi_f = peaceman_solve(
        Pf_f[:, :, PERF], Tf_f[:, :, PERF], kf_f[:, :, PERF],
        rf_f[:, :, PERF], q_f, rw, z_datum, prod_mask,
    )
    # Pbhp = Pavg - sign * q / (c * wi_sum)  => fit c per type
    z_index = np.arange(2, 12)
    Ph_f = hydrostatic(Pf_f[:, :, PERF], rf_f[:, :, PERF], z_datum, z_index)
    wi_sum_f = layer_wi(kf_f[:, :, PERF], Tf_f[:, :, PERF], rw).sum(axis=2)
    Pavg_f = (layer_wi(kf_f[:, :, PERF], Tf_f[:, :, PERF], rw) * Ph_f).sum(axis=2) / np.maximum(wi_sum_f, 1e-12)
    for label, sl in [("prod", slice(0, N_PROD)), ("inj", slice(N_PROD, 9)), ("all", slice(0, 9))]:
        sign = np.where(prod_mask[sl], 1.0, -1.0)
        # bhp = Pavg - sign * q / (c * wi) => c = sign * q / ((Pavg - bhp) * wi)
        denom = (Pavg_f[:, :, sl] - bhp_f[:, :, sl]) * wi_sum_f[:, :, sl]
        numer = sign * q_f[:, :, sl]
        m = np.abs(denom) > 1e-6
        c = np.median(numer[m] / denom[m])
        print(f"  {label} median WI scale c={c:.4f}")
        Pbhp_c = Pavg_f[:, :, sl] - sign * q_f[:, :, sl] / np.maximum(c * wi_sum_f[:, :, sl], 1e-12)
        report(f"  fit {label} scaled WI", Pbhp_c, bhp_f[:, :, sl])

    # apply producer c and injector c on test
    Ph_t = hydrostatic(Pf_t[:, :, PERF], rf_t[:, :, PERF], z_datum, z_index)
    wi_sum_t = layer_wi(kf_t[:, :, PERF], Tf_t[:, :, PERF], rw).sum(axis=2)
    Pavg_t = (layer_wi(kf_t[:, :, PERF], Tf_t[:, :, PERF], rw) * Ph_t).sum(axis=2) / np.maximum(wi_sum_t, 1e-12)
    cs = {}
    for label, sl in [("prod", slice(0, N_PROD)), ("inj", slice(N_PROD, 9))]:
        sign = np.where(prod_mask[sl], 1.0, -1.0)
        denom = (Pavg_f[:, :, sl] - bhp_f[:, :, sl]) * wi_sum_f[:, :, sl]
        numer = sign * q_f[:, :, sl]
        m = np.abs(denom) > 1e-6
        cs[label] = np.median(numer[m] / denom[m])
    Pbhp_scaled = np.empty_like(bhp_t)
    Pbhp_scaled[:, :, :N_PROD] = Pavg_t[:, :, :N_PROD] - q_t[:, :, :N_PROD] / np.maximum(cs["prod"] * wi_sum_t[:, :, :N_PROD], 1e-12)
    Pbhp_scaled[:, :, N_PROD:] = Pavg_t[:, :, N_PROD:] + q_t[:, :, N_PROD:] / np.maximum(cs["inj"] * wi_sum_t[:, :, N_PROD:], 1e-12)
    print(f"\nTEST scaled WI c_prod={cs['prod']:.4f} c_inj={cs['inj']:.4f}")
    report("scaled WI all", Pbhp_scaled, bhp_t)
    report("scaled WI prod", Pbhp_scaled[:, :, :N_PROD], bhp_t[:, :, :N_PROD])
    report("scaled WI inj", Pbhp_scaled[:, :, N_PROD:], bhp_t[:, :, N_PROD:])

    print("\n=== ENERGY: layer-allocated vs mean (test open) ===")
    q_p_t = q_t[:, :, :N_PROD]
    q_p_f = q_f[:, :, :N_PROD]
    e_tt = e_t
    e_ff = e_f
    open_t = q_p_t > 1.0
    open_f = q_p_f > 1.0
    print("energy when shut max", np.abs(e_tt[q_p_t < 1e-6]).max())

    Tmean_t = Tf_t[:, :, PERF, :N_PROD].mean(axis=2)
    rmean_t = rf_t[:, :, PERF, :N_PROD].mean(axis=2)
    Pmean_t = Pf_t[:, :, PERF, :N_PROD].mean(axis=2)
    Tmean_f = Tf_f[:, :, PERF, :N_PROD].mean(axis=2)
    rmean_f = rf_f[:, :, PERF, :N_PROD].mean(axis=2)

    # layer rates from Peaceman, producers only (q_layer > 0 out of reservoir)
    _, q_layer_t, _ = peaceman_solve(
        Pf_t[:, :, PERF], Tf_t[:, :, PERF], kf_t[:, :, PERF],
        rf_t[:, :, PERF], q_t, rw, z_datum, prod_mask,
    )
    qL = q_layer_t[:, :, :, :N_PROD]  # (N,t,Z,7)  sum over Z should ~ q
    print("  mean |sum_z q_layer - q| prod", np.mean(np.abs(qL.sum(axis=2) - q_p_t)))

    h_cp_layer = CP_WATER * Tf_t[:, :, PERF, :N_PROD]
    E_layer_cp = (qL * rf_t[:, :, PERF, :N_PROD] * h_cp_layer).sum(axis=2)
    E_mean_cp = q_p_t * rmean_t * CP_WATER * Tmean_t
    E_sc_cp = q_p_t * RHO_SC * CP_WATER * Tmean_t
    h_iapws_t = iapws97_region1_enthalpy_jkg(Tmean_t, Pmean_t)
    print("  mean IAPWS h J/kg", h_iapws_t[open_t].mean())

    def fit_scale(unit, tgt, mask):
        x, y = unit[mask], tgt[mask]
        return float(np.dot(x, y) / np.dot(x, x))

    print("\n-- unscaled --")
    report("q rho cp T mean", E_mean_cp, e_tt, open_t)
    report("q_layer rho cp T", E_layer_cp, e_tt, open_t)
    report("q rho_sc cp T", E_sc_cp, e_tt, open_t)

    E_mean_cp_f = q_p_f * rmean_f * CP_WATER * Tmean_f
    E_sc_cp_f = q_p_f * RHO_SC * CP_WATER * Tmean_f
    s1 = fit_scale(E_mean_cp_f, e_ff, open_f)
    s2 = fit_scale(E_sc_cp_f, e_ff, open_f)
    print("\n-- one scale from fit --")
    report(f"q rho cp T * {s1:.6g}", E_mean_cp * s1, e_tt, open_t)
    report(f"q rho_sc cp T * {s2:.6g}", E_sc_cp * s2, e_tt, open_t)

    # thermodynamic enthalpy for exponential EOS
    # h = cp (T-Tref) + (1 - beta T_abs) (P-Pref) / rho
    def h_eos(T, P, rho, cp=CP_WATER, Tref=25.0, Pref=100.0):
        T_abs = T + 273.15
        return cp * (T - Tref) + (1.0 - beta * T_abs) * (P - Pref) * 1000.0 / np.maximum(rho, 1e-6)

    h_e = h_eos(Tmean_t, Pmean_t, rmean_t)
    h_e_f = h_eos(Tmean_f, Pf_f[:, :, PERF, :N_PROD].mean(axis=2), rmean_f)
    E_eos = q_p_t * rmean_t * h_e
    E_eos_f = q_p_f * rmean_f * h_e_f
    s3 = fit_scale(E_eos_f, e_ff, open_f)
    print("\n-- EOS enthalpy --")
    report("q rho h_eos unscaled", E_eos, e_tt, open_t)
    report(f"q rho h_eos * {s3:.6g}", E_eos * s3, e_tt, open_t)
    E_sc_eos = q_p_t * RHO_SC * h_e
    E_sc_eos_f = q_p_f * RHO_SC * h_e_f
    s4 = fit_scale(E_sc_eos_f, e_ff, open_f)
    report("q rho_sc h_eos unscaled", E_sc_eos, e_tt, open_t)
    report(f"q rho_sc h_eos * {s4:.6g}", E_sc_eos * s4, e_tt, open_t)

    # Fit cp, Tref in E = q * rho_sc * cp * (T - Tref)
    # E = A * q*rho_sc*T + B * q*rho_sc
    X1 = (q_p_f * RHO_SC * Tmean_f)[open_f]
    X0 = (q_p_f * RHO_SC)[open_f]
    y = e_ff[open_f]
    A = np.stack([X1, X0], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    cp_hat, href = coef[0], coef[1]
    # href = -cp * Tref => Tref = -href/cp
    Tref_hat = -href / cp_hat
    print(f"\n  lstsq E=q*rho_sc*(cp T + h0): cp={cp_hat:.6g} h0={href:.6g} Tref={Tref_hat:.4f}")
    pred = q_p_t * RHO_SC * (cp_hat * Tmean_t + href)
    report("q rho_sc (cp T + h0)", pred, e_tt, open_t)

    X1r = (q_p_f * rmean_f * Tmean_f)[open_f]
    X0r = (q_p_f * rmean_f)[open_f]
    Ar = np.stack([X1r, X0r], axis=1)
    coefr, *_ = np.linalg.lstsq(Ar, y, rcond=None)
    print(f"  lstsq E=q*rho_res*(cp T + h0): cp={coefr[0]:.6g} h0={coefr[1]:.6g} Tref={-coefr[1]/coefr[0]:.4f}")
    pred_r = q_p_t * rmean_t * (coefr[0] * Tmean_t + coefr[1])
    report("q rho_res (cp T + h0)", pred_r, e_tt, open_t)

    # add pressure
    X2 = (q_p_f * RHO_SC * Pf_f[:, :, PERF, :N_PROD].mean(axis=2))[open_f]
    A3 = np.stack([X1, X0, X2], axis=1)
    c3, *_ = np.linalg.lstsq(A3, y, rcond=None)
    print(f"  lstsq E=q*rho_sc*(a T + b + c P): a={c3[0]:.6g} b={c3[1]:.6g} c={c3[2]:.6g}")
    pred3 = q_p_t * RHO_SC * (
        c3[0] * Tmean_t + c3[1] + c3[2] * Pmean_t
    )
    report("q rho_sc (aT+b+cP)", pred3, e_tt, open_t)

    print("\n=== Per-well energy MAPE for q*rho_sc*(cp T + h0) ===")
    for w in range(N_PROD):
        m = q_p_t[:, :, w] > 1
        p = pred[:, :, w][m]
        g = e_tt[:, :, w][m]
        mape = np.mean(np.abs(p - g) / np.maximum(g, 1)) * 100
        rel = np.linalg.norm(p - g) / np.linalg.norm(g)
        print(f"  {WELL_NAMES[w]} MAPE={mape:.4f}% relL2={rel:.5f}")

    print("\n=== Is energy exact to float32 of some formula? ===")
    for name, p in [
        ("q rho cp T", E_mean_cp),
        ("scaled q rho_sc cp T", E_sc_cp * s2),
        ("lstsq q rho_sc (cpT+h0)", pred),
        ("lstsq q rho_res (cpT+h0)", pred_r),
        ("lstsq q rho_sc (aT+b+cP)", pred3),
    ]:
        rel = np.abs(p - e_tt) / np.maximum(e_tt, 1)
        rel = rel[open_t]
        print(
            f"  {name:32s}  median rel={np.median(rel):.4e}  "
            f"p99 rel={np.percentile(rel,99):.4e}  "
            f"frac<1e-4={(rel<1e-4).mean():.4f}  frac<1e-3={(rel<1e-3).mean():.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
