"""Train and evaluate fraud models: XGBoost and LightGBM vs a logistic-regression baseline.

Protocol (all choices are made on the validation slice; the test slice is touched once):
  1. clean -> engineer features -> chronological 60/20/20 split
  2. grid: model x imbalance strategy x feature set, early stopping on val PR-AUC
  3. pick the best config per model by val PR-AUC
  4. pick the decision threshold on val (max F1)
  5. refit the chosen configs with 5 seeds, report test metrics as mean +/- std,
     plus a bootstrap CI on test PR-AUC (the test slice has only ~75 frauds)

Run:  python train.py            (writes results/ and figures/)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (ENGINEERED_FEATURES, RAW_FEATURES, add_features, load_and_clean,
                          time_split)

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
FEATURE_SETS = {"raw": RAW_FEATURES, "raw+engineered": RAW_FEATURES + ENGINEERED_FEATURES}
IMBALANCE = ["none", "class_weight", "sqrt_weight", "undersample_1:20"]


# ---------------------------------------------------------------- imbalance handling
def prepare_train(X: pd.DataFrame, y: pd.Series, strategy: str, seed: int):
    """Returns X, y, sample_weight for the chosen imbalance strategy."""
    ratio = (y == 0).sum() / (y == 1).sum()
    if strategy == "none":
        return X, y, None
    if strategy == "class_weight":  # same as scale_pos_weight = neg/pos
        return X, y, np.where(y == 1, ratio, 1.0)
    if strategy == "sqrt_weight":  # softer re-weighting; often better calibrated
        return X, y, np.where(y == 1, np.sqrt(ratio), 1.0)
    if strategy.startswith("undersample"):
        k = int(strategy.split(":")[1])
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y.to_numpy() == 1)
        neg = np.flatnonzero(y.to_numpy() == 0)
        keep = np.concatenate([pos, rng.choice(neg, size=min(len(neg), k * len(pos)), replace=False)])
        keep.sort()
        return X.iloc[keep], y.iloc[keep], None
    raise ValueError(strategy)


# ---------------------------------------------------------------- models
def fit_xgb(Xtr, ytr, wtr, Xva, yva, seed):
    m = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=200, random_state=seed, n_jobs=-1,
    )
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.best_iteration + 1


def fit_lgbm(Xtr, ytr, wtr, Xva, yva, seed):
    m = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
    return m, m.best_iteration_


def fit_logreg(Xtr, ytr, wtr, Xva, yva, seed):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(Xtr.fillna(0), ytr)
    return m, None


FITTERS = {"XGBoost": fit_xgb, "LightGBM": fit_lgbm}


def proba(model, X):
    # trees handle missing values natively; the linear baseline needs them filled
    is_linear = not isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier))
    return model.predict_proba(X.fillna(0) if is_linear else X)[:, 1]


# ---------------------------------------------------------------- metrics
def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def metrics_at(y, p, thr, amount=None):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat),
        "accuracy": (tp + tn) / len(y),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": thr,
        **({"fraud_amount_caught_pct": float(100 * amount[(y == 1) & (yhat == 1)].sum()
                                             / amount[y == 1].sum())} if amount is not None else {}),
    }


def bootstrap_pr_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def recall_at_precision(y, p, target):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec[prec >= target]
    return float(ok.max()) if len(ok) else 0.0


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    df = add_features(load_and_clean(str(ROOT / "data" / "creditcard.csv")))
    tr, va, te = time_split(df)
    summary = {
        "rows_after_cleaning": len(df),
        "duplicates_dropped": df.attrs.get("duplicates_dropped"),
        "split": {name: {"rows": len(s), "frauds": int(s.Class.sum()),
                         "fraud_rate_pct": round(100 * s.Class.mean(), 3)}
                  for name, s in [("train", tr), ("val", va), ("test", te)]},
    }
    ytr, yva, yte = tr.Class, va.Class, te.Class
    print(json.dumps(summary, indent=1))

    # ---- 1. grid search on validation
    grid = []
    for model_name, fit in FITTERS.items():
        for fs_name, cols in FEATURE_SETS.items():
            for strat in IMBALANCE:
                Xt, yt, w = prepare_train(tr[cols], ytr, strat, seed=0)
                m, n_trees = fit(Xt, yt, w, va[cols], yva, seed=0)
                pva = proba(m, va[cols])
                row = {"model": model_name, "features": fs_name, "imbalance": strat,
                       "val_pr_auc": average_precision_score(yva, pva),
                       "val_roc_auc": roc_auc_score(yva, pva), "trees": n_trees}
                grid.append(row)
                print(f"{model_name:9s} {fs_name:15s} {strat:17s} val PR-AUC {row['val_pr_auc']:.4f}  trees {n_trees}")
    grid_df = pd.DataFrame(grid).sort_values("val_pr_auc", ascending=False)
    grid_df.to_csv(RES / "grid_validation.csv", index=False)

    # ---- 2. final configs (best per model on val), multi-seed test evaluation
    final, curves, test_probs = {}, {}, {}
    for model_name, fit in FITTERS.items():
        best = grid_df[grid_df.model == model_name].iloc[0]
        cols = FEATURE_SETS[best.features]
        runs, pte_seeds = [], []
        for seed in SEEDS:
            Xt, yt, w = prepare_train(tr[cols], ytr, best.imbalance, seed)
            m, _ = fit(Xt, yt, w, va[cols], yva, seed)
            thr = best_f1_threshold(yva, proba(m, va[cols]))  # threshold chosen on VAL
            pte = proba(m, te[cols])
            r = metrics_at(yte.to_numpy(), pte, thr, te.Amount.to_numpy())
            r["recall_at_p90"] = recall_at_precision(yte, pte, 0.90)
            runs.append(r)
            pte_seeds.append(pte)
            if seed == 0:
                model0 = m
        pte_mean = np.mean(pte_seeds, axis=0)
        test_probs[model_name] = pte_mean
        agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
               for k in runs[0]}
        lo, hi = bootstrap_pr_auc(yte, pte_seeds[0])
        final[model_name] = {"config": {"features": best.features, "imbalance": best.imbalance},
                             "test": agg, "test_pr_auc_95ci_seed0": [lo, hi], "per_seed": runs}
        curves[model_name] = precision_recall_curve(yte, pte_seeds[0])
        # feature importance (gain) for the seed-0 model
        if model_name == "XGBoost":
            imp = pd.Series(model0.get_booster().get_score(importance_type="gain")).sort_values()
        else:
            imp = pd.Series(model0.booster_.feature_importance("gain"), index=cols).sort_values()
        imp.tail(20).plot.barh(figsize=(6, 6), color="#3b6ea5")
        plt.title(f"{model_name}: top 20 features by gain")
        plt.tight_layout()
        plt.savefig(FIG / f"importance_{model_name.lower()}.png", dpi=120)
        plt.close()

    # ---- 3. baseline: logistic regression (class-balanced) and "always legit"
    cols = FEATURE_SETS["raw+engineered"]
    lr, _ = fit_logreg(tr[cols], ytr, None, va[cols], yva, 0)
    thr = best_f1_threshold(yva, proba(lr, va[cols]))
    plr = proba(lr, te[cols])
    final["LogisticRegression"] = {"config": {"features": "raw+engineered", "imbalance": "class_weight"},
                                   "test": {k: {"mean": float(v), "std": 0.0} for k, v in
                                            {**metrics_at(yte.to_numpy(), plr, thr, te.Amount.to_numpy()),
                                             "recall_at_p90": recall_at_precision(yte, plr, 0.9)}.items()}}
    curves["LogisticRegression"] = precision_recall_curve(yte, plr)
    final["AlwaysLegit"] = {"test": {"accuracy": {"mean": float(1 - yte.mean()), "std": 0.0},
                                     "recall": {"mean": 0.0, "std": 0.0},
                                     "pr_auc": {"mean": float(yte.mean()), "std": 0.0}}}

    # ---- 4. what a random split would have claimed (the leakage story)
    best = grid_df[grid_df.model == "XGBoost"].iloc[0]
    cols = FEATURE_SETS[best.features]
    rtr, rtmp = train_test_split(df, test_size=0.4, stratify=df.Class, random_state=0)
    rva, rte = train_test_split(rtmp, test_size=0.5, stratify=rtmp.Class, random_state=0)
    Xt, yt, w = prepare_train(rtr[cols], rtr.Class, best.imbalance, 0)
    m, _ = fit_xgb(Xt, yt, w, rva[cols], rva.Class, 0)
    final["XGBoost_random_split"] = {"test_pr_auc": float(average_precision_score(rte.Class, proba(m, rte[cols])))}

    # ---- plots
    plt.figure(figsize=(6.5, 5))
    for name, (p, r, _) in curves.items():
        ap = final[name]["test"]["pr_auc"]["mean"]
        plt.plot(r, p, label=f"{name} (PR-AUC {ap:.3f})")
    plt.axhline(yte.mean(), ls="--", c="grey", lw=1, label=f"chance ({yte.mean():.4f})")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of alerts that are fraud)")
    plt.title("Precision-recall on the held-out test period")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "pr_curves.png", dpi=120)
    plt.close()

    # threshold trade-off for the best model
    top = max(FITTERS, key=lambda k: final[k]["test"]["pr_auc"]["mean"])
    p, r, t = precision_recall_curve(yte, test_probs[top])
    plt.figure(figsize=(6.5, 4))
    plt.plot(t, p[:-1], label="precision")
    plt.plot(t, r[:-1], label="recall")
    plt.axvline(final[top]["test"]["threshold"]["mean"], ls="--", c="grey", lw=1,
                label="threshold picked on validation")
    plt.xscale("logit")
    plt.xlabel("decision threshold")
    plt.title(f"{top}: precision / recall vs threshold (test)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "threshold_tradeoff.png", dpi=120)
    plt.close()

    summary["final"] = final
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (RES / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v.get("test", v) for k, v in final.items()}, indent=1, default=float)[:4000])
    print("done in", summary["runtime_sec"], "s")


if __name__ == "__main__":
    main()
