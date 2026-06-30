import torch
import torch.nn as nn

from config import *


class ReconstructionHead(nn.Module):

    """
    G operator

    Reconstruct future market field
    from latent trajectory
    """

    def __init__(self):

        super().__init__()

        self.reconstruct = nn.Sequential(

            nn.Linear(

                LATENT_DIM,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                NUM_ASSETS * FEATURE_DIM

            )

        )

    def forward(self,
                x):

        #
        # input shape
        #
        # [batch,
        #  trajectory_length,
        #  latent_dim]
        #

        batch_size = x.shape[0]

        #
        # flatten time
        #

        x = x.view(

            -1,

            LATENT_DIM

        )

        #
        # reconstruct
        #

        x = self.reconstruct(
            x
        )

        #
        # restore full shape
        #

        x = x.view(

            batch_size,

            TRAJECTORY_LENGTH,

            NUM_ASSETS,

            FEATURE_DIM

        )

        return x
