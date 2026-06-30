import torch
import torch.nn as nn

from src.config import *


class ObservationEncoder(nn.Module):

    """
    R operator

    Converts raw market observations
    into latent state vectors
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                FEATURE_DIM,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                LATENT_DIM
            )

        )

    def forward(self,
                x):

        #
        # x shape:
        #
        # [batch_size,
        #  num_assets,
        #  feature_dim]
        #

        batch_size = x.shape[0]

        #
        # flatten asset dimension
        #

        x = x.view(

            -1,

            FEATURE_DIM

        )

        #
        # encode each asset
        #

        x = self.encoder(
            x
        )

        #
        # restore shape
        #

        x = x.view(

            batch_size,

            NUM_ASSETS,

            LATENT_DIM

        )

        return x
