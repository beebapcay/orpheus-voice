#!/usr/bin/env python
"""Compute EER, minDCF and actDCF from a score file.

Score convention used everywhere in this script: HIGHER = MORE LIKELY REAL.

Accepted inputs
---------------
1. AntiDeepfake score file (written by `main.py inference`):
       ID,Score,Label
       X-001,"[3.43, -3.55]",0
   `Score` is "[fake_logit, real_logit]", `Label` is 1 = real, 0 = fake.
   The score used is the log-odds  real_logit - fake_logit.
   (softmax P(real) is a monotonic function of it, so the EER is the same,
   but log-odds does not saturate at 0/1 and can be compared with an LLR
   threshold.)

2. Any CSV with one numeric score column and one label column:
       python evaluate.py scores.csv --score-col score --label-col label
   Labels may be real/bonafide/genuine/1 or fake/spoof/deepfake/0.
   If the file uses "higher = more likely fake", add --higher-is-fake.

Metrics
-------
* EER: same algorithm as ASVspoof / AntiDeepfake evaluation.py
  (sweep thresholds, take the point where |FRR - FAR| is smallest,
  report the mean of the two), so numbers are comparable with the repo.
* DCF(t) = beta * Pmiss(t) + Pfa(t), normalised so that "accept everything"
  costs 1.  beta = (Cmiss * (1 - p_spoof)) / (Cfa * p_spoof).
  ASVspoof 5 defaults: Cmiss = 1, Cfa = 10, p_spoof = 0.05  ->  beta = 1.9.
* minDCF: DCF at the best threshold chosen after seeing the labels.
* actDCF: DCF at the fixed Bayes threshold t = -ln(beta) (about -0.64).
  Only meaningful if the scores are calibrated LLRs. Raw log-odds from an
  uncalibrated model usually give actDCF > minDCF.

Breakdowns
----------
--prefix P1 P2   : metrics on rows whose ID starts with "P1-" (real and fake),
                   same as AntiDeepfake evaluation.py.
--protocol CSV --group-col Attack
                 : join the protocol on ID; for each value of the group
                   column among FAKE rows, compute metrics of
                   (all real rows) vs (fake rows of that group).
                   Use this for per-generator (per-attack) EER.

Examples
--------
    python evaluate.py Log/exps/exp_x/evaluation_score.csv
    python evaluate.py score.csv --protocol protocols/test.csv --group-col Attack
    python evaluate.py --selftest

Requires only numpy and pandas; runs on Python >= 3.8.
"""
import argparse
import ast
import math
import sys

import numpy as np
import pandas as pd

REAL_NAMES = {"real", "bonafide", "bona-fide", "bona_fide", "genuine", "1", "true"}
FAKE_NAMES = {"fake", "spoof", "deepfake", "synthetic", "0", "false"}


# ----------------------------------------------------------------------
# Core metrics
# ----------------------------------------------------------------------
def det_curve(real_scores, fake_scores):
    """FRR (Pmiss) and FAR (Pfa) for every threshold, ASVspoof style.

    A trial is accepted as real when score > threshold.  The first point
    (threshold below every score) is FRR = 0, FAR = 1.
    """
    real_scores = np.asarray(real_scores, dtype=np.float64)
    fake_scores = np.asarray(fake_scores, dtype=np.float64)
    n_real, n_fake = real_scores.size, fake_scores.size
    all_scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]
    real_rejected = np.cumsum(labels)
    fake_accepted = n_fake - (np.arange(1, all_scores.size + 1) - real_rejected)
    frr = np.concatenate([[0.0], real_rejected / n_real])
    far = np.concatenate([[1.0], fake_accepted / n_fake])
    thresholds = np.concatenate([[all_scores[order[0]] - 0.001], all_scores[order]])
    return frr, far, thresholds


def compute_eer(real_scores, fake_scores):
    frr, far, thresholds = det_curve(real_scores, fake_scores)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0), float(thresholds[i])


def dcf_beta(c_miss=1.0, c_fa=10.0, p_spoof=0.05):
    return (c_miss * (1.0 - p_spoof)) / (c_fa * p_spoof)


def compute_min_dcf(real_scores, fake_scores, beta):
    frr, far, thresholds = det_curve(real_scores, fake_scores)
    dcf = beta * frr + far
    i = int(np.argmin(dcf))
    return float(dcf[i]), float(thresholds[i])


def compute_act_dcf(real_scores, fake_scores, beta):
    """DCF at the Bayes threshold -ln(beta); accept as real if score >= t."""
    t = -math.log(beta)
    p_miss = float(np.mean(np.asarray(real_scores) < t))
    p_fa = float(np.mean(np.asarray(fake_scores) >= t))
    return beta * p_miss + p_fa, t


def all_metrics(real_scores, fake_scores, beta):
    real_scores = np.asarray(real_scores, dtype=np.float64)
    fake_scores = np.asarray(fake_scores, dtype=np.float64)
    if real_scores.size == 0 or fake_scores.size == 0:
        return None
    eer, eer_thr = compute_eer(real_scores, fake_scores)
    min_dcf, _ = compute_min_dcf(real_scores, fake_scores, beta)
    act_dcf, _ = compute_act_dcf(real_scores, fake_scores, beta)
    return {
        "n_real": int(real_scores.size),
        "n_fake": int(fake_scores.size),
        "eer_%": 100.0 * eer,
        "eer_threshold": eer_thr,
        "minDCF": min_dcf,
        "actDCF": act_dcf,
    }


# ----------------------------------------------------------------------
# Reading score files
# ----------------------------------------------------------------------
def _parse_label(value):
    s = str(value).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s in REAL_NAMES:
        return 1
    if s in FAKE_NAMES:
        return 0
    raise ValueError("Unrecognised label: {!r}".format(value))


def _antideepfake_logodds(cell):
    pair = ast.literal_eval(str(cell))
    if len(pair) != 2:
        raise ValueError("Expected [fake_logit, real_logit], got {!r}".format(cell))
    fake_logit, real_logit = float(pair[0]), float(pair[1])
    return real_logit - fake_logit


def load_scores(path, score_col=None, label_col="Label", id_col="ID",
                higher_is_fake=False):
    """Return a DataFrame with columns ID, score (higher = real), label (1/0)."""
    df = pd.read_csv(path)
    if score_col is None:
        score_col = "Score"
    for col in (id_col, score_col, label_col):
        if col not in df.columns:
            raise KeyError("Column {!r} not in {} (columns: {})".format(
                col, path, list(df.columns)))

    first = str(df[score_col].iloc[0]).strip()
    if first.startswith("["):
        score = df[score_col].apply(_antideepfake_logodds)
    else:
        score = pd.to_numeric(df[score_col], errors="raise").astype(float)
    if higher_is_fake:
        score = -score

    out = pd.DataFrame({
        "ID": df[id_col].astype(str).str.strip(),
        "score": score.values,
        "label": df[label_col].apply(_parse_label).values,
    })
    if not np.all(np.isfinite(out["score"].values)):
        bad = out.loc[~np.isfinite(out["score"].values), "ID"].tolist()[:5]
        raise ValueError("Non-finite scores, e.g. {}".format(bad))
    if out["ID"].duplicated().any():
        dup = out.loc[out["ID"].duplicated(), "ID"].tolist()[:5]
        raise ValueError("Duplicated IDs, e.g. {}".format(dup))
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def build_report(df, beta, prefixes=None, groups=None):
    rows = []

    def add(name, sub):
        m = all_metrics(sub.loc[sub.label == 1, "score"],
                        sub.loc[sub.label == 0, "score"], beta)
        if m is None:
            print("[skip] {}: needs both real and fake rows".format(name),
                  file=sys.stderr)
            return
        m["subset"] = name
        rows.append(m)

    add("Pooled", df)

    for p in prefixes or []:
        add("prefix:" + p, df[df.ID.str.startswith(p + "-")])

    if groups is not None:
        real = df[df.label == 1]
        fake = df[df.label == 0]
        for g in sorted(fake["group"].dropna().astype(str).unique()):
            add("group:" + g, pd.concat([real, fake[fake.group.astype(str) == g]]))

    cols = ["subset", "n_real", "n_fake", "eer_%", "minDCF", "actDCF", "eer_threshold"]
    return pd.DataFrame(rows)[cols]


def selftest():
    beta = dcf_beta()
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print("{:<48s} got {:.4f}  want {:.4f}  {}".format(
            name, got, want, "OK" if good else "FAIL"))

    # 8-utterance example: real {0.9,0.7,0.6,0.2}, fake {0.8,0.4,0.3,0.1}
    real = [0.9, 0.7, 0.6, 0.2]
    fake = [0.8, 0.4, 0.3, 0.1]
    m = all_metrics(real, fake, beta)
    check("beta (ASVspoof 5)", beta, 1.9)
    check("toy EER (%)", m["eer_%"], 25.0)
    check("toy minDCF", m["minDCF"], 0.725)
    check("toy actDCF, probability scores", m["actDCF"], 1.0)

    # EER/minDCF ignore monotonic transforms; actDCF does not.
    rng = np.random.RandomState(0)
    r = rng.normal(1.0, 1.0, 2000)
    f = rng.normal(-1.0, 1.0, 3000)
    m1 = all_metrics(r, f, beta)
    m2 = all_metrics(3 * r + 7, 3 * f + 7, beta)
    check("EER invariant to 3*s+7", m2["eer_%"], m1["eer_%"])
    check("minDCF invariant to 3*s+7", m2["minDCF"], m1["minDCF"])

    # Brute-force EER: minimise |FRR-FAR| over all thresholds.
    thr = np.concatenate([[-np.inf], np.sort(np.concatenate([r, f]))])
    frr = np.array([np.mean(r <= t) for t in thr])
    far = np.array([np.mean(f > t) for t in thr])
    i = np.argmin(np.abs(frr - far))
    check("EER vs brute force (%)", m1["eer_%"], 100 * (frr[i] + far[i]) / 2)
    check("minDCF vs brute force", m1["minDCF"], float(np.min(beta * frr + far)))

    # Two Gaussians with means +-1, sd 1: theoretical EER = Phi(-1) = 15.87 %
    check("Gaussian EER near 15.87 % (tol 1.5)", m1["eer_%"], 15.87, tol=1.5)

    # Swapped direction must give EER near 100 - EER.
    m3 = all_metrics(-r, -f, beta)
    check("flipped scores: EER near 100-EER (tol 0.1)",
          m3["eer_%"], 100 - m1["eer_%"], tol=0.1)

    # AntiDeepfake score string -> log-odds
    check("parse '[3.0, -1.0]' -> real-fake", _antideepfake_logodds("[3.0, -1.0]"), -4.0)

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score_csv", nargs="?")
    ap.add_argument("--score-col", default=None,
                    help="score column (default: Score)")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--higher-is-fake", action="store_true",
                    help="flip the sign: the file uses higher = more likely fake")
    ap.add_argument("--prefix", nargs="*", default=None,
                    help="ID prefixes (text before the first '-') to report separately")
    ap.add_argument("--protocol", default=None,
                    help="protocol CSV to join on ID for per-group metrics")
    ap.add_argument("--group-col", default="Attack",
                    help="protocol column used for per-group metrics (default: Attack)")
    ap.add_argument("--c-miss", type=float, default=1.0)
    ap.add_argument("--c-fa", type=float, default=10.0)
    ap.add_argument("--p-spoof", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="save the table as CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.score_csv:
        ap.error("score_csv is required (or use --selftest)")

    beta = dcf_beta(args.c_miss, args.c_fa, args.p_spoof)
    df = load_scores(args.score_csv, args.score_col, args.label_col,
                     args.id_col, args.higher_is_fake)

    groups = None
    if args.protocol:
        proto = pd.read_csv(args.protocol, dtype=str)
        if args.group_col not in proto.columns:
            raise KeyError("Column {!r} not in protocol".format(args.group_col))
        proto = proto[[args.id_col, args.group_col]].rename(
            columns={args.id_col: "ID", args.group_col: "group"})
        proto["ID"] = proto["ID"].str.strip()
        df = df.merge(proto, on="ID", how="left")
        missing = int(df["group"].isna().sum())
        if missing:
            print("[warn] {} score rows have no match in the protocol".format(missing),
                  file=sys.stderr)
        groups = True

    report = build_report(df, beta, args.prefix, groups)
    print("Score: higher = real. beta = {:.3f}, actDCF threshold = {:.3f}".format(
        beta, -math.log(beta)))
    print("EER in %, lower is better. minDCF/actDCF: lower is better, 1.0 = no better than accepting everything.")
    with pd.option_context("display.width", 160, "display.max_rows", 500):
        print(report.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    if args.out:
        report.to_csv(args.out, index=False)
        print("Saved to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
