import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    variance = np.var(x, axis=-1, keepdims =True)
    mean = np.mean(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_hat + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    Q_proj = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    scores = Q_proj @ K_proj.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    attn_output = attn_weights @ V_proj
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = attn_output @ W_o
    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, np.dot(x,W1) + b1)
    output = np.dot(hidden, W2) + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attention_output = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    x_hat = layer_norm(x + attention_output, gamma1, beta1)
    ffn_output = feed_forward(x_hat, W1, b1, W2, b2)
    output = layer_norm(x_hat + ffn_output, gamma2, beta2)
    return output