import torch
import torch.nn as nn

from config import *


class DecisionFunctional(nn.Module):

    """
    D operator

    Convert future trajectory
    into trade decision outputs
    """

    def __init__(self):

        super().__init__()

        self.decision = nn.Sequential(

            nn.Linear(

                LATENT_DIM * TRAJECTORY_LENGTH,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                4

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
        # flatten trajectory
        #

        x = x.view(

            batch_size,

            -1

        )

        output = self.decision(
            x
        )

        #
        # constrain probability
        #

        output[:, 1] = torch.sigmoid(

            output[:, 1]

        )

        return output
