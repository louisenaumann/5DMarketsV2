import torch
import torch.nn as nn

from config import *


class DVFCorrelation(nn.Module):

    """
    M_A operator

    Correlation deformation field

    Assets influence one another
    through learned correlation structure
    """

    def __init__(self):

        super().__init__()

        #
        # learn interaction weights
        #

        self.attention = nn.MultiheadAttention(

            embed_dim=LATENT_DIM,

            num_heads=4,

            batch_first=True

        )

    def forward(self,
                x):

        #
        # x shape
        #
        # [batch,
        #  num_assets,
        #  latent_dim]
        #

        output, _ = self.attention(

            x,

            x,

            x

        )

        return output
