import torch
import torch.nn as nn

from config import *


class CompressionLayer(nn.Module):

    """
    C operator

    Compress full market state
    before temporal memory
    """

    def __init__(self):

        super().__init__()

        self.compress = nn.Sequential(

            nn.Linear(

                LATENT_DIM * NUM_ASSETS,

                512

            ),

            nn.ReLU(),

            nn.Linear(

                512,

                256

            )

        )

    def forward(self,
                x):

        #
        # input:
        #
        # [batch,3200]
        #

        output = self.compress(
            x
        )

        return output
