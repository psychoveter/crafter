# how to prepare data? explore crafter scene generation process


"""
# 1. Create autoencoder for onehot state representation
# 2.


0. State tensor S
1. VAE S -> S
2. Action generative model (A, S) -> S on top of VAE somehow
3. S* -- space-time
    3.1 Structure of time embeds behaviour
    3.2 It requires learning on logic changes
    3.3 What to do with open environments?
    3.4 Here search is for additive, one-shot S* models
4. Conditional VAE
"""




# Ideas to learn encoders
# 1. Try UNet architecture
#     https://www.geeksforgeeks.org/u-net-architecture-explained/
#     https://github.com/gerardrbentley/Pytorch-U-Net-AutoEncoder/blob/master/models.py
#
# 2. Try L2 norm on hidden layer
# 3. Setup tSNE to show latent space
# 4. Setup learning with Ray on AWS and meta optimization
