import torch
import torch.nn as nn

from config import *


class TemporalMemory(nn.Module):

    """
    H operator

    Historical memory operator

    Encodes temporal dependence
    in market deformation history
    """

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size = 256, #LATENT_DIM * NUM_ASSETS,

            hidden_size= 256, #LATENT_DIM * NUM_ASSETS,

            batch_first=True

        )

    def forward(self,
                x):

        #
        # input shape
        #
        # [batch,
        #  sequence,
        #  latent_dim]
        #

        output, _ = self.lstm(
            x
        )

        #
        # return final hidden state
        #

        return output[:, -1, :]
