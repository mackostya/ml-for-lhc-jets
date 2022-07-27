from networks.CNN import cnn_model
import numpy as np

def test_cnn():
    # Test if the shape bottled down to a classification one hot encoding
    input = np.ones((1,40,40,1))
    one_hot_shape = (1,2)
    model = cnn_model()
    output = model(input)
    assert one_hot_shape == output.shape, f"The output shape {output.shape} is not compatible with the classification one hot encoding {one_hot_shape}"