import torch
import torch.nn as nn

from config import *


class DVFCapitalFlow(nn.Module):

    """
    M_C operator

    Capital flow deformation field

    Capital leaving one asset
    reallocates elsewhere
    """

    def __init__(self):

        super().__init__()

        #
        # allocation scoring
        #

        self.scorer = nn.Linear(

            LATENT_DIM,

            1

        )

        #
        # transform flow effect
        #

        self.project = nn.Linear(

            1,

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
        # compute allocation score
        #

        scores = self.scorer(
            x
        )

        #
        # normalize capital distribution
        #

        weights = torch.softmax(

            scores,

            dim=1

        )

        #
        # enforce conservation
        #

        mean_weight = torch.mean(

            weights,

            dim=1,

            keepdim=True

        )

        net_flow = (

            weights

            -

            mean_weight

        )

        #
        # project flow back to latent space
        #

        flow_effect = self.project(

            net_flow

        )

        output = (

            x

            +

            flow_effect

        )

        return output
