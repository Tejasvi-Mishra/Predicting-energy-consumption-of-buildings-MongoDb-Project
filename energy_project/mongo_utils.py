"""MongoDB connection helper (PyMongo)."""
import os

DB_NAME = "energy_db"
COLLECTION = "energy_readings"
RESULTS_COLLECTION = "model_results"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

_fallback_db = None


def get_db():
    """Return a PyMongo database. Falls back to an in-memory store only if no
    MongoDB server is reachable (set REQUIRE_MONGO=1 to forbid the fallback)."""
    global _fallback_db
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        print(f"[mongo] Connected to MongoDB at {MONGO_URI}")
        return client[DB_NAME]
    except Exception as exc:
        if os.environ.get("REQUIRE_MONGO") == "1":
            raise
        print(f"[mongo] MongoDB unavailable ({type(exc).__name__}); "
              "using in-memory fallback store.")
        from mini_mongo import Database
        if _fallback_db is None:
            _fallback_db = Database()
        return _fallback_db
