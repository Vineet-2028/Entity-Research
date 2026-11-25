"""Small synthetic CPU demonstration; not the competition pipeline or leaderboard."""
import argparse
import json
import platform
from difflib import SequenceMatcher
from pathlib import Path
import lightgbm as lgb
import numpy as np

SEED = 20260926


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def dataset(start, stop, seed):
    rng = np.random.default_rng(seed)
    rows, labels, groups = [], [], []
    for q in range(start, stop):
        name = f"{rng.choice(['cedar', 'maple', 'oak', 'elm'])} {q} trading"
        address = int(rng.integers(10, 900))
        singleton = q % 5 == 0
        for d in range(3):
            positive = d == 0 and not singleton
            if positive:
                other = name.replace("trading", "trdg") if q % 2 else name
                number = address if q % 7 else address + 1
            elif d == 0:  # plausible singleton near-copy
                other, number = name, address + 12
            elif d == 1:  # near-copy, often identical name
                other, number = name, address + int(rng.integers(2, 40))
            else:
                other, number = f"{rng.choice(['pine', 'ash', 'fir'])} {q+1} services", address
            rows.append([similarity(name, other), float(address == number), abs(address-number)])
            labels.append(int(positive))
            groups.append(q)
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


def macro_f05_reference(y, pred, groups):
    scores = []
    for group in np.unique(groups):
        a, b = y[groups == group], pred[groups == group]
        tp = int(np.sum((a == 1) & (b == 1)))
        nt, npred = int(a.sum()), int(b.sum())
        scores.append(1.0 if nt == npred == 0 else 1.25 * tp / (0.25 * nt + npred))
    return float(np.mean(scores))


def run():
    train, val, test = (dataset(a,b,SEED+i) for i,(a,b) in enumerate([(0,200),(200,260),(260,340)]))
    assert not (set(train[2]) & set(val[2]) or set(val[2]) & set(test[2]) or set(train[2]) & set(test[2]))
    model = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, learning_rate=0.1,
                               random_state=SEED, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
    model.fit(train[0], train[1])
    vp = model.booster_.predict(val[0])
    candidates = np.linspace(0.05,0.95,19)
    threshold = max(candidates, key=lambda t: macro_f05_reference(val[1], vp >= t, val[2]))
    x,y,g = test
    pred = model.booster_.predict(x) >= threshold
    baseline = (x[:,0] >= 0.85) & (x[:,1] == 1)
    return {"scope":"SYNTHETIC_CPU_DEMO_NOT_COMPETITION", "seed":SEED,
            "train_groups":200,"validation_groups":60,"test_groups":80,
            "threshold_from_validation":float(threshold),
            "validation_macro_f05":macro_f05_reference(val[1], vp >= threshold,val[2]),
            "test_baseline_macro_f05":macro_f05_reference(y,baseline,g),
            "test_lightgbm_macro_f05":macro_f05_reference(y,pred,g),
            "python":platform.python_version(), "numpy":np.__version__,"lightgbm":lgb.__version__}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=json.dumps(run(),indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result,encoding="utf-8")
    print(result,end="")
