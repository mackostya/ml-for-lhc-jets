from networks.AE import ae_model
import numpy as np

def test_autoencoder():
    # Test if the shape is reconstructed after the bottleneck
    input = np.ones((1,40,40,1))
    model = ae_model()
    output = model(input)
    assert output.shape == input.shape