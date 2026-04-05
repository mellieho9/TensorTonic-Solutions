def retraining_policy(daily_stats, config):
    days_since_retrain = 0
    remaining_budget = config["budget"]
    last_retrain_day = 0
    days_retrained = []

    for stat in daily_stats:
        # Step 4: Increment first
        days_since_retrain += 1

        # Step 1: Check triggers
        should_retrain = (
            stat["drift_score"] > config["drift_threshold"]
            or stat["performance"] < config["performance_threshold"]
            or days_since_retrain >= config["max_staleness"]
        )

        # Step 2 & 3: Check constraints and retrain
        if should_retrain:
            can_retrain = (
                (stat["day"] - last_retrain_day >= config["cooldown"] or stat["day"] == 1)
                and remaining_budget > 0
            )
            if can_retrain:
                remaining_budget -= config["retrain_cost"]
                last_retrain_day = stat["day"]
                days_since_retrain = 0
                days_retrained.append(stat["day"])

    return days_retrained