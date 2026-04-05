import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        z1 = x @ self.W1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2
        
        # 2. Shortcut Path: Project x to match out_channels
        shortcut = x @ self.Ws  # Used to match dimensions
        
        # 3. Add & Final ReLU
        # Final output = ReLU(F(x) + shortcut)
        return np.maximum(0, z2 + shortcut)
