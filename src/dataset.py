import torch
from torch.utils.data import Dataset
import numpy as np

from config import *


class MarketDataset(Dataset):

    def __init__(self,
                 num_samples=5000):

        self.samples = []

        for _ in range(num_samples):

            #
            # market field
            #
            # shape:
            # [num_assets, feature_dim]
            #

            current_state = np.random.normal(

                0,

                1,

                (

                    NUM_ASSETS,

                    FEATURE_DIM

                )

            )

            #
            # synthetic future trajectory
            #

            future_trajectory = []

            state = current_state.copy()

            for t in range(
                    TRAJECTORY_LENGTH):

                #
                # small deformation
                #

                deformation = np.random.normal(

                    0,

                    0.05,

                    (

                        NUM_ASSETS,

                        FEATURE_DIM

                    )

                )

                state = (
                    state
                    +
                    deformation
                )

                future_trajectory.append(
                    state.copy()
                )

            future_trajectory = np.array(
                future_trajectory
            )

            self.samples.append(

                (

                    current_state,

                    future_trajectory

                )

            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self,
                    idx):

        current_state, future = self.samples[idx]

        return (

            torch.tensor(
                current_state,
                dtype=torch.float32
            ),

            torch.tensor(
                future,
                dtype=torch.float32
            )
        )
