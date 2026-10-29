# Predicting Energy Consumption of Buildings — MongoDB + PyMongo + Machine Learning

## Project structure
| File | Purpose |
|---|---|
| `generate_dataset.py` | Creates the synthetic dataset `data/energy_data.csv` (6,015 rows, incl. missing values & duplicates) |
| `mongo_utils.py` | PyMongo connection helper (`MONGO_URI`, DB `energy_db`) |
| `load_data.py` | Inserts the CSV into MongoDB, creates indexes, runs example queries / aggregations |
| `train_models.py` | Reads from MongoDB -> preprocessing -> trains & evaluates ML models -> saves plots/results |
| `run_all.py` | Runs the whole pipeline end to end |
| `mini_mongo.py` | In-memory stand-in used ONLY if no MongoDB server is reachable |
| `outputs/` | Metrics (`results.json`), query output, and figures |

## How to run
1. Install and start MongoDB (Community Server) — default `mongodb://localhost:27017`.
2. `pip install -r requirements.txt`
3. `python run_all.py`
   (or step by step: `python generate_dataset.py`, `python load_data.py`, `python train_models.py`)

To use a different server: `export MONGO_URI="mongodb://host:27017"`.
To forbid the in-memory fallback (recommended for submission screenshots): `export REQUIRE_MONGO=1`.

Verify in `mongosh`:
```
use energy_db
db.energy_readings.countDocuments()
db.model_results.find().pretty()
```

## Models & metrics
* Regression (target `energy_kwh`): Linear Regression, Random Forest Regressor -> R², MAE, RMSE
* Classification (target High/Normal usage): Logistic Regression, Decision Tree, Random Forest -> Accuracy, Precision, Recall, F1
