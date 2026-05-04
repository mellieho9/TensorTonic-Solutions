def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    lookup rules
    - look up offline features
    - user not found -> provided defaults
    - combine offline + online features -> return one combined dict in the same order as input
    """
    def join_features(online_features, lookup_features):
        for feature in lookup_features.keys():
            if feature not in online_features:
                online_features[feature] = lookup_features[feature]
    feature_vectors = []
    for request in requests:
        online_features = request["online_features"]
        if request["user_id"] not in feature_store:
            join_features(online_features, defaults)
        else:
            join_features(online_features, feature_store[request["user_id"]])
        feature_vectors.append(online_features)
    return feature_vectors