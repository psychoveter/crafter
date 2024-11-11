import torch

class OneHotEncoder(torch.nn.Module):
    """
    I need to convert from image to one hot representation of the image
    for some datasets where I don't have semantic map
    """
    def __init__(self):
        super(OneHotEncoder, self).__init__()