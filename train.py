import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import core as sc
from . import registry

VAL0, TEST0, PHYS0 = 500, 1000, 5000
GRID = [dict(n_estimators=n, max_depth=d) for n in (100, 300) for d in (None, 8, 16)]


def dataset(seeds, cond, wl_t, mmode, weighted, task, meth):
    Xs, ys, gs = [], [], []
    for s in seeds:
        d = sc.build((s,), tuple(cond), wl_t, mmode, weighted)
        X, y = sc.subset(d, task, meth)
        if len(y):
            Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


def fit_best(Xtr, ytr, Xva, yva):
    best = None
    for hp in GRID:
        m = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced", **hp).fit(Xtr, ytr)
        f = macro_f1(yva, m.predict(Xva))
        if best is None or f > best[0]:
            best = (f, hp, m)
    return best


def rates(y, p, prob):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(accuracy=(tp + tn) / max(len(y), 1), macro_f1=macro_f1(y, p),
                sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                auc=roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan)


def boot_ci(groups, stat, B=1000, seed=0):
    """95% CI by resampling whole fields (cells within a field are correlated,
    so resampling individual cells would give intervals that are too narrow)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    v = [stat(np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])) for _ in range(B)]
    return tuple(float(x) for x in np.nanpercentile(v, [2.5, 97.5]))


def evaluate(task, models, wl_t, mmode, weighted, n_test):
    rows, deltas = [], []
    seeds = range(TEST0, TEST0 + n_test)
    for name, cond in sc.COND_SET.items():
        pred, prob, ys, g = {}, {}, {}, None
        for m in sc.METHODS:
            X, y, g = dataset(seeds, cond, wl_t, mmode, weighted, task, m)
            pred[m] = models[m].predict(X); prob[m] = models[m].predict_proba(X)[:, 1]; ys[m] = y
        y = ys["unmix"]
        assert all(np.array_equal(y, ys[m]) for m in sc.METHODS), "cell order differs between methods"
        for m in sc.METHODS:
            lo, hi = boot_ci(g, lambda ix, p=pred[m]: macro_f1(y[ix], p[ix]))
            rows.append(dict(condition=name, method=m, n_fields=len(np.unique(g)), n_cells=len(y),
                             **rates(y, pred[m], prob[m]), f1_lo=lo, f1_hi=hi))
        # paired comparison: does unmixing beat plain flat-fielded RGB, on the same resampled fields?
        d = lambda ix: macro_f1(y[ix], pred["unmix"][ix]) - macro_f1(y[ix], pred["odrgb"][ix])
        lo, hi = boot_ci(g, d)
        deltas.append(dict(condition=name, unmix_minus_odrgb=d(np.arange(len(y))), ci_lo=lo, ci_hi=hi))
    return rows, deltas


def physics_checks(wl_t, mmode, weighted, n=10, k=5.0):
    """Validates the UNTRAINED stages (registration, unmixing, residual anomaly flag) against simulator ground truth."""
    wl, cond, ri = np.array(wl_t, float), sc.Cond(), sc.ref_index(wl_t)
    cr = sc.clean_residual(tuple(cond), wl_t, mmode, weighted)
    thr = cr.mean() + k * cr.std()
    keep = [i for i in range(len(wl)) if i != ri]
    reg, corr, rec, fpr = [], [[] for _ in range(sc.K)], [], []
    for s in range(PHYS0, PHYS0 + n):
        a = sc.acquire(sc.make_field(s, defects=True), wl, cond, s)
        P = sc.run_pipeline(a, sc.get_M(mmode, cond, wl), weighted)
        reg.append(np.sqrt(np.mean((P["shifts"][keep] + a["true_shift"][keep]) ** 2)))
        for j in range(sc.K):
            corr[j].append(np.corrcoef(P["C"][j].ravel(), a["true_C"][j].ravel())[0, 1])
        flag, dm = P["res"] > thr, a["dmask"]
        rec.append((flag & dm).sum() / max(dm.sum(), 1)); fpr.append((flag & ~dm).sum() / (~dm).sum())
    return dict(registration_rmse_px=float(np.mean(reg)),
                concentration_pearson_r={s: float(np.nanmean(c)) for s, c in zip(sc.STAINS, corr)},
                defect_recall=float(np.mean(rec)), defect_false_positive_rate=float(np.mean(fpr)), threshold_k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="malaria", choices=list(sc.TASKS))
    ap.add_argument("--leds", default="ext", choices=["ext", "base"])
    ap.add_argument("--mmode", default="calibrated", choices=["literature", "calibrated"])
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--no-weight", action="store_true")
    a = ap.parse_args()
    assert a.n_train <= VAL0 and a.n_val <= TEST0 - VAL0, "split ranges would overlap"

    wl_t = tuple(int(w) for w in (sc.EXT_WL if a.leds == "ext" else sc.BASE_WL))
    weighted, base = not a.no_weight, sc.Cond()

    models, hps = {}, {}
    for m in sc.METHODS:
        Xtr, ytr, _ = dataset(range(a.n_train), base, wl_t, a.mmode, weighted, a.task, m)
        Xva, yva, _ = dataset(range(VAL0, VAL0 + a.n_val), base, wl_t, a.mmode, weighted, a.task, m)
        f, hps[m], models[m] = fit_best(Xtr, ytr, Xva, yva)
        print(f"{m:6s} val macro-F1 {f:.3f}  {hps[m]}  (train cells {len(ytr)}, val cells {len(yva)})")

    rows, deltas = evaluate(a.task, models, wl_t, a.mmode, weighted, a.n_test)
    phys = physics_checks(wl_t, a.mmode, weighted)
    cr = sc.clean_residual(tuple(base), wl_t, a.mmode, weighted)

    show = ["condition", "method", "n_fields", "accuracy", "macro_f1", "f1_lo", "f1_hi", "sensitivity", "specificity", "auc"]
    print("\nTEST (zero-shot on every non-baseline condition)")
    print(pd.DataFrame(rows)[show].round(3).to_string(index=False))
    print("\nPAIRED: unmixed minus flat-fielded RGB (macro-F1). CI excluding 0 = a real difference")
    print(pd.DataFrame(deltas).round(3).to_string(index=False))
    print("\nPHYSICS CHECKS (untrained stages)"); print(phys)

    meta = dict(data_source="simulator", classes=sc.TASKS[a.task]["classes"], wavelengths=list(wl_t),
                mmode=a.mmode, weighted=weighted, hyperparameters=hps,
                splits=dict(train=[0, a.n_train], val=[VAL0, VAL0 + a.n_val], test=[TEST0, TEST0 + a.n_test], unit="field"),
                test_results=rows, paired_deltas=deltas, physics_checks=phys,
                clean_residual=dict(mean=float(cr.mean()), std=float(cr.std())),
                core_sha256=registry.file_sha256(Path(sc.__file__)))
    print("\nSaved model:", registry.save_model(a.task, models, meta))


if __name__ == "__main__":
    main()
