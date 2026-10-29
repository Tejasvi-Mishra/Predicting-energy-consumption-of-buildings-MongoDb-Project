"""Generate a synthetic building energy-consumption dataset.

Columns: energy readings + weather + occupancy + building attributes.
The dataset is synthetic (no public data is bundled) but follows realistic
physical patterns: HVAC load grows with distance from a 21 C comfort point,
occupancy drives plug/lighting load, and each building type has its own
daily schedule and weekend behaviour.
Deliberately includes ~2% missing values and a few duplicate records so the
preprocessing stage has real work to do.
"""
import numpy as np
import pandas as pd

SEED = 42
N_BUILDINGS = 24
READINGS_PER_BUILDING = 250

# type -> (energy intensity kWh per 100 m2, weekend factor)
TYPES = {
    "Office":      (14.0, 0.55),
    "Residential": (9.0, 1.15),
    "Retail":      (16.0, 1.10),
    "School":      (11.0, 0.40),
    "Hospital":    (24.0, 0.95),
}


def hour_profile(btype, hour):
    h = np.asarray(hour)
    if btype in ("Office", "School"):
        return np.clip(np.exp(-((h - 13) ** 2) / 28), 0.08, 1)
    if btype == "Retail":
        return np.clip(np.exp(-((h - 15) ** 2) / 30), 0.10, 1)
    if btype == "Residential":
        return np.clip(0.35 + 0.65 * np.exp(-((h - 20) ** 2) / 14), 0.2, 1)
    return 0.75 + 0.25 * np.sin((h - 6) / 24 * 2 * np.pi)  # Hospital


def main(path="data/energy_data.csv"):
    rng = np.random.default_rng(SEED)
    type_names = list(TYPES)
    rows = []
    rid = 1
    for b in range(1, N_BUILDINGS + 1):
        btype = type_names[(b - 1) % len(type_names)]
        area = float(rng.integers(800, 9000))
        year_built = int(rng.integers(1960, 2021))
        intensity, wk = TYPES[btype]
        ts = pd.to_datetime("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 365 * 24, READINGS_PER_BUILDING), unit="h")
        for t in ts:
            month, hour, dow = t.month, t.hour, t.dayofweek
            weekend = int(dow >= 5)
            temp = (24 + 12 * np.sin((month - 4) / 12 * 2 * np.pi)
                    + 4 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 2.5))
            humidity = float(np.clip(60 - 0.8 * (temp - 24) + rng.normal(0, 12), 15, 100))
            wind = float(np.clip(rng.gamma(2.0, 2.0), 0, 25))
            solar = float(max(0, 800 * np.sin(np.pi * (hour - 6) / 12)) * rng.uniform(0.3, 1.0)) \
                if 6 <= hour <= 18 else 0.0
            cond = ("Rainy" if humidity > 85 and rng.random() < 0.6 else
                    "Foggy" if humidity > 80 and rng.random() < 0.3 else
                    "Clear" if solar > 450 else "Cloudy")
            prof = hour_profile(btype, hour)
            wfac = wk if weekend else 1.0
            occupancy = int(max(0, area / 12 * prof * wfac * rng.uniform(0.8, 1.1)))
            base = area / 100 * intensity * (0.45 + 0.55 * float(prof)) * wfac
            hvac = 0.11 * abs(temp - 21) ** 1.3 * area / 100
            people = 0.12 * occupancy
            solar_gain = 0.002 * solar * area / 100 * (1 if temp > 24 else 0.3)
            age = 1 + (2025 - year_built) / 400
            energy = (base + hvac + people + solar_gain) * age
            energy = max(1.0, energy * rng.normal(1, 0.07))
            rows.append(dict(record_id=rid, building_id=f"B{b:03d}", building_type=btype,
                             floor_area=area, year_built=year_built, timestamp=t,
                             temperature=round(temp, 2), humidity=round(humidity, 1),
                             wind_speed=round(wind, 2), solar_radiation=round(solar, 1),
                             weather_condition=cond, occupancy=occupancy,
                             is_weekend=weekend, energy_kwh=round(energy, 2)))
            rid += 1
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # inject ~2% missing values and a few duplicate records
    for col in ("temperature", "humidity", "occupancy"):
        idx = rng.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df[col] = df[col].astype("float64")
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=SEED)], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
