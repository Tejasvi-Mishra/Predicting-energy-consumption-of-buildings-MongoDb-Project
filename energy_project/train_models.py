"""Step 2: Read data from MongoDB, preprocess, train and evaluate ML models.

Regression      -> predict energy_kwh            (Linear Regression, Random Forest)
Classification  -> predict High / Normal usage   (Logistic Regression, Decision Tree, Random Forest)
Metrics: R2, MAE, RMSE | Accuracy, Precision, Recall, F1
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mongo_utils import get_db, COLLECTION, RESULTS_COLLECTION

SEED = 42
NUM = ["floor_area", "building_age", "temperature", "humidity", "wind_speed",
       "solar_radiation", "occupancy", "is_weekend", "hour", "month", "temp_deviation"]
CAT = ["building_type", "weather_condition", "season"]


# ---------------------------------------------------------------- load
def load_from_mongo(db):
    docs = list(db[COLLECTION].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    print(f"Loaded {len(df)} documents from MongoDB")
    return df


# ---------------------------------------------------------------- preprocess
def preprocess(df):
    log = {"rows_loaded": len(df)}
    log["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    before = len(df)
    df = df.drop_duplicates(subset="record_id").copy()
    log["duplicates_removed"] = before - len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["building_age"] = 2025 - df["year_built"]
    df["season"] = df["month"].map(lambda m: "Winter" if m in (12, 1, 2) else
                                   "Spring" if m in (3, 4, 5) else
                                   "Summer" if m in (6, 7, 8) else "Autumn")
    # comfort deviation uses temperature, so impute it first (median per month)
    df["temperature"] = df["temperature"].fillna(df.groupby("month")["temperature"].transform("median"))
    df["temp_deviation"] = (df["temperature"] - 21).abs()

    # classification target: top third of consumption = "High"
    thr = df["energy_kwh"].quantile(2 / 3)
    df["high_usage"] = (df["energy_kwh"] > thr).astype(int)
    log["high_usage_threshold_kwh"] = round(float(thr), 2)
    log["rows_after_cleaning"] = len(df)
    return df, log


def make_preprocessor(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),   # categorical encoding
    ])


# ---------------------------------------------------------------- models
def run_regression(df):
    X, y = df[NUM + CAT], df["energy_kwh"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    models = {
        "Linear Regression": Pipeline([("pre", make_preprocessor(True)), ("m", LinearRegression())]),
        "Random Forest Regressor": Pipeline([("pre", make_preprocessor(False)),
                                             ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                         random_state=SEED, n_jobs=-1))]),
    }
    res, preds = {}, {}
    for name, p in models.items():
        p.fit(Xtr, ytr)
        yp = p.predict(Xte)
        preds[name] = yp
        res[name] = {"R2": r2_score(yte, yp), "MAE": mean_absolute_error(yte, yp),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, yp)))}
        print(f"[Regression] {name:<24} R2={res[name]['R2']:.4f}  MAE={res[name]['MAE']:.2f}  RMSE={res[name]['RMSE']:.2f}")
    return res, models, (Xte, yte, preds)


def run_classification(df):
    X, y = df[NUM + CAT], df["high_usage"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    models = {
        "Logistic Regression": Pipeline([("pre", make_preprocessor(True)),
                                         ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([("pre", make_preprocessor(False)),
                                   ("m", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED))]),
        "Random Forest Classifier": Pipeline([("pre", make_preprocessor(False)),
                                              ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                           random_state=SEED, n_jobs=-1))]),
    }
    res, cms = {}, {}
    for name, p in models.items():
        p.fit(Xtr, ytr)
        yp = p.predict(Xte)
        res[name] = {"Accuracy": accuracy_score(yte, yp), "Precision": precision_score(yte, yp),
                     "Recall": recall_score(yte, yp), "F1": f1_score(yte, yp)}
        cms[name] = confusion_matrix(yte, yp)
        r = res[name]
        print(f"[Classification] {name:<24} Acc={r['Accuracy']:.4f}  P={r['Precision']:.4f}  "
              f"R={r['Recall']:.4f}  F1={r['F1']:.4f}")
    return res, models, cms


# ---------------------------------------------------------------- plots
def make_plots(df, reg_res, reg_models, reg_data, clf_res, clf_models, cms):
    Xte, yte, preds = reg_data
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for a, (name, yp) in zip(ax, preds.items()):
        a.scatter(yte, yp, s=6, alpha=0.4, color="#2b6cb0")
        lim = [0, max(yte.max(), yp.max())]
        a.plot(lim, lim, "r--", lw=1)
        a.set_title(f"{name}\nR²={reg_res[name]['R2']:.3f}, MAE={reg_res[name]['MAE']:.1f}")
        a.set_xlabel("Actual energy (kWh)")
    ax[0].set_ylabel("Predicted energy (kWh)")
    plt.tight_layout(); plt.savefig("outputs/actual_vs_predicted.png", dpi=150); plt.close()

    rf = reg_models["Random Forest Regressor"]
    names = rf.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names).sort_values().tail(12)
    plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="#2f855a")
    plt.title("Random Forest – top 12 feature importances"); plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, (name, cm) in zip(ax, cms.items()):
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["Normal", "High"]); a.set_yticks([0, 1], ["Normal", "High"])
        a.set_title(name, fontsize=9); a.set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    plt.tight_layout(); plt.savefig("outputs/confusion_matrices.png", dpi=150); plt.close()

    avg = df.groupby("building_type")["energy_kwh"].mean().sort_values()
    plt.figure(figsize=(6, 3.6)); avg.plot.barh(color="#dd6b20")
    plt.xlabel("Average energy (kWh)"); plt.title("Average consumption by building type")
    plt.tight_layout(); plt.savefig("outputs/avg_energy_by_type.png", dpi=150); plt.close()

    plt.figure(figsize=(6.5, 3.6))
    h = df.groupby("hour")["energy_kwh"].mean(); plt.plot(h.index, h.values, marker="o", color="#805ad5")
    plt.xlabel("Hour of day"); plt.ylabel("Average energy (kWh)"); plt.title("Daily consumption profile")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("outputs/hourly_profile.png", dpi=150); plt.close()


def main():
    db = get_db()
    df = load_from_mongo(db)
    df, log = preprocess(df)
    print("Preprocessing log:", json.dumps(log, indent=2))
    reg_res, reg_models, reg_data = run_regression(df)
    clf_res, clf_models, cms = run_classification(df)
    make_plots(df, reg_res, reg_models, reg_data, clf_res, clf_models, cms)

    out = {"preprocessing": log, "regression": reg_res, "classification": clf_res,
           "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
           "train_test_sizes": {"total": len(df), "train": int(len(df) * .8), "test": len(df) - int(len(df) * .8)}}
    with open("outputs/results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # store the evaluation results back in MongoDB
    rc = db[RESULTS_COLLECTION]
    rc.delete_many({})
    rc.insert_many([{"task": "regression", "model": k, **{m: float(v) for m, v in r.items()}} for k, r in reg_res.items()] +
                   [{"task": "classification", "model": k, **{m: float(v) for m, v in r.items()}} for k, r in clf_res.items()])
    print("Results saved to outputs/results.json and MongoDB collection", RESULTS_COLLECTION)


if __name__ == "__main__":
    main()
