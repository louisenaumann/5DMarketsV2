import torch
import torch.nn as nn

from config import *


class FutureTransition(nn.Module):

    """
    Φ operator

    Predict future market trajectory
    from historical market state
    """

    def __init__(self):

        super().__init__()

        self.project = nn.Sequential(

            nn.Linear(

                256, #LATENT_DIM * NUM_ASSETS,

                256 #512

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                LATENT_DIM * TRAJECTORY_LENGTH

            )

        )

    def forward(self,
                x):

        #
        # input:
        #
        # [batch, latent_dim]
        #

        output = self.project(
            x
        )

        #
        # reshape
        #

        output = output.view(

            -1,

            TRAJECTORY_LENGTH,

            LATENT_DIM

        )

        return output
