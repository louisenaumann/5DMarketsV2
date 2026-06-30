import torch
import torch.nn as nn

from config import *


class DVFSector(nn.Module):

    """
    M_B operator

    Sector deformation field

    Assets inside same economic sector
    influence one another
    """

    def __init__(self):

        super().__init__()

        #
        # transform sector interactions
        #

        self.linear = nn.Linear(

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

        #
        # divide into sectors
        #

        num_sectors = 10

        assets_per_sector = 5

        sectors = []

        for i in range(
                num_sectors):

            start = i * assets_per_sector

            end = (
                start
                +
                assets_per_sector
            )

            sector = x[
                :,
                start:end,
                :
            ]

            #
            # sector average state
            #

            sector_mean = torch.mean(

                sector,

                dim=1,

                keepdim=True

            )

            #
            # broadcast sector influence
            #

            sector = (
                sector
                +
                sector_mean
            )

            sectors.append(
                sector
            )

        output = torch.cat(

            sectors,

            dim=1

        )

        output = self.linear(
            output
        )

        return output
