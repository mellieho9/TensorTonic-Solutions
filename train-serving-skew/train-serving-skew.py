import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    results = {}
    
    for feature in train_dist:
        # Convert lists to numpy arrays
        p_train = np.array(train_dist[feature]) + eps
        p_serving = np.array(serving_dist[feature]) + eps
        
        # Calculate PSI: sum( (P_serving - P_train) * ln(P_serving / P_train) )
        psi_val = np.sum((p_serving - p_train) * np.log(p_serving / p_train))
        
        # Determine if skewed based on the threshold
        results[feature] = {
            "psi": float(psi_val),
            "skewed": bool(psi_val >= threshold)
        }
        
    return results