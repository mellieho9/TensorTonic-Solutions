def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    production_accuracy = calc_accuracy(production_log)
    shadow_accuracy = calc_accuracy(shadow_log)
    accuracy_gain = shadow_accuracy - production_accuracy
    
    shadow_latency = calc_latency(shadow_log)

    agreement_rate = calc_agreement(production_log, shadow_log)

    promote = accuracy_gain >= criteria["min_accuracy_gain"] and shadow_latency <= criteria["max_latency_p95"] and agreement_rate >= criteria["min_agreement_rate"]

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency,
            "agreement_rate": agreement_rate
        }
    }

def calc_accuracy(log):
    n = len(log)
    accurate_predictions = sum(1 for req in log if req["actual"] == req["prediction"])
    return accurate_predictions / n

def calc_latency(log):
    latencies = [req["latency_ms"] for req in log]
    latencies.sort()
    rank = math.ceil(0.95 * len(latencies))
    return latencies[rank  - 1]

def calc_agreement(prod, shadow):
    prod_dict = {req["input_id"]: req["prediction"] for req in prod}
    shadow_dict = {req["input_id"]: req["prediction"] for req in shadow}
    
    common_ids = set(prod_dict.keys()) & set(shadow_dict.keys())
    if not common_ids: return 0
    
    agreement_count = sum(1 for id in common_ids if prod_dict[id] == shadow_dict[id])
    return agreement_count / len(common_ids)

    