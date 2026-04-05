from datetime import datetime

def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    registry = sorted(
        models,
        key=lambda x: (-x["accuracy"], x["latency"], -datetime.strptime(x["timestamp"], "%Y-%m-%d").timestamp())
    )
    return registry[0]["name"]