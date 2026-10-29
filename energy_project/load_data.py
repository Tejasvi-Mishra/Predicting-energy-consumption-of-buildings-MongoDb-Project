"""Step 1: Load the CSV into MongoDB with PyMongo and run example queries."""
import io
import contextlib
import numpy as np
import pandas as pd
from mongo_utils import get_db, COLLECTION

CSV = "data/energy_data.csv"


def load(db):
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    # NaN -> None so MongoDB stores a proper null
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    for r in records:
        r["timestamp"] = pd.Timestamp(r["timestamp"]).to_pydatetime()
        for k, v in r.items():
            if isinstance(v, (np.integer,)): r[k] = int(v)
            elif isinstance(v, (np.floating,)): r[k] = float(v)

    col = db[COLLECTION]
    col.delete_many({})                       # idempotent re-runs
    col.insert_many(records)
    col.create_index([("building_id", 1)])
    col.create_index([("timestamp", 1)])
    col.create_index([("building_type", 1)])
    return col


def queries(col):
    print("=== MongoDB queries ===")
    print("Total documents:", col.count_documents({}))
    print("Documents with missing temperature:", col.count_documents({"temperature": None}))
    print("Weekend documents:", col.count_documents({"is_weekend": 1}))
    print("High consumption (> 1500 kWh):", col.count_documents({"energy_kwh": {"$gt": 1500}}))

    print("\nSample document:")
    print(col.find_one({}, {"_id": 0}))

    print("\nAverage energy (kWh) by building type:")
    for r in col.aggregate([
        {"$group": {"_id": "$building_type", "avg_kwh": {"$avg": "$energy_kwh"},
                    "readings": {"$sum": 1}}},
        {"$sort": {"avg_kwh": -1}}]):
        print(f"  {r['_id']:<12} avg={r['avg_kwh']:8.1f}  n={r['readings']}")

    print("\nAverage energy by weather condition:")
    for r in col.aggregate([
        {"$group": {"_id": "$weather_condition", "avg_kwh": {"$avg": "$energy_kwh"}}},
        {"$sort": {"avg_kwh": -1}}]):
        print(f"  {r['_id']:<8} avg={r['avg_kwh']:8.1f}")

    print("\nAverage energy by hour of day (top 5 peak hours):")
    for r in col.aggregate([
        {"$group": {"_id": "$hour", "avg_kwh": {"$avg": "$energy_kwh"}}},
        {"$sort": {"avg_kwh": -1}}, {"$limit": 5}]):
        print(f"  hour {r['_id']:>2}  avg={r['avg_kwh']:8.1f}")

    print("\nTop 5 buildings by total consumption:")
    for r in col.aggregate([
        {"$group": {"_id": "$building_id", "total_kwh": {"$sum": "$energy_kwh"}}},
        {"$sort": {"total_kwh": -1}}, {"$limit": 5}]):
        print(f"  {r['_id']}  total={r['total_kwh']:,.0f}")


def main():
    db = get_db()
    col = load(db)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        queries(col)
    text = buf.getvalue()
    print(text)
    with open("outputs/mongodb_query_results.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
