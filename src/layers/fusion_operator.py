import torch
import torch.nn as nn

from config import *


class FusionOperator(nn.Module):

    """
    F operator

    Fuse multiple deformation fields
    into unified market deformation field
    """

    def __init__(self):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Linear(

                LATENT_DIM * 4,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                LATENT_DIM

            )

        )

    def forward(self,
                m_a,
                m_b,
                m_c,
                m_d):

        #
        # concatenate fields
        #

        x = torch.cat(

            [

                m_a,

                m_b,

                m_c,

                m_d

            ],

            dim=2

        )

        #
        # shape:
        #
        # [batch,
        #  num_assets,
        #  256]
        #

        output = self.fusion(
            x
        )

        return output
