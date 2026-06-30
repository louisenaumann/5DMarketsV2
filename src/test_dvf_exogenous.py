import torch
import torch.nn as nn

from config import *


class DVFExogenous(nn.Module):

    """
    M_D operator

    Exogenous shock field

    External events inject force
    into market system
    """

    def __init__(self):

        super().__init__()

        #
        # transform shock vector
        #

        self.shock_transform = nn.Linear(

            LATENT_DIM,

            LATENT_DIM

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

        batch_size = x.shape[0]

        #
        # generate global shock
        #

        shock = torch.randn(

            batch_size,

            1,

            LATENT_DIM

        )

        #
        # transform shock
        #

        shock = self.shock_transform(
            shock
        )

        #
        # broadcast to all assets
        #

        shock = shock.repeat(

            1,

            NUM_ASSETS,

            1

        )

        #
        # inject shock
        #

        output = (

            x

            +

            shock

        )

        return output
