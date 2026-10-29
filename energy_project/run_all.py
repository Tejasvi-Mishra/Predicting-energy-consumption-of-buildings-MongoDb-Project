"""Run the complete pipeline: generate data -> load into MongoDB -> train/evaluate.
Note: in-memory fallback is process-local, so everything runs in one process here."""
import generate_dataset, load_data, train_models
import mongo_utils

generate_dataset.main()
load_data.main()
train_models.main()
