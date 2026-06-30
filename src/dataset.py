import torch
from torch.utils.data import Dataset
import numpy as np

from config import *


class MarketDataset(Dataset):

    def __init__(self,
                 num_samples=5000):

        self.samples = []

        num_sectors = 10
        assets_per_sector = 5

        for _ in range(num_samples):

            #
            # initial market state
            #

            current_state = np.random.normal(

                0,

                1,

                (

                    NUM_ASSETS,

                    FEATURE_DIM

                )

            )

            future_trajectory = []

            state = current_state.copy()

            #
            # evolve through time
            #

            for t in range(
                    TRAJECTORY_LENGTH):

                ##################################
                # A — correlation deformation
                ##################################

                correlation_noise = np.random.normal(

                    0,

                    0.01,

                    (

                        NUM_ASSETS,

                        FEATURE_DIM

                    )

                )

                #
                # correlate neighboring assets
                #

                for i in range(

                        1,

                        NUM_ASSETS):

                    correlation_noise[i] += (

                        0.5

                        *

                        correlation_noise[i - 1]

                    )

                ##################################
                # B — sector deformation
                ##################################

                sector_effect = np.zeros(

                    (

                        NUM_ASSETS,

                        FEATURE_DIM

                    )

                )

                for s in range(
                        num_sectors):

                    sector_shift = np.random.normal(

                        0,

                        0.02

                    )

                    start = s * assets_per_sector

                    end = start + assets_per_sector

                    sector_effect[
                        start:end
                    ] += sector_shift

                ##################################
                # C — capital flow
                ##################################

                capital_flow = np.zeros(

                    (

                        NUM_ASSETS,

                        FEATURE_DIM

                    )

                )

                sector_a = np.random.randint(
                    0,
                    num_sectors
                )

                sector_b = np.random.randint(
                    0,
                    num_sectors
                )

                flow = np.random.normal(

                    0,

                    0.03

                )

                a_start = sector_a * assets_per_sector

                a_end = a_start + assets_per_sector

                b_start = sector_b * assets_per_sector

                b_end = b_start + assets_per_sector

                capital_flow[
                    a_start:a_end
                ] += flow

                capital_flow[
                    b_start:b_end
                ] -= flow

                ##################################
                # D — exogenous shock
                ##################################

                exogenous = np.zeros(

                    (

                        NUM_ASSETS,

                        FEATURE_DIM

                    )

                )

                if np.random.rand() < 0.1:

                    shock = np.random.normal(

                        0,

                        0.05

                    )

                    exogenous += shock

                ##################################
                # total deformation
                ##################################

                total_deformation = (

                    correlation_noise

                    +

                    sector_effect

                    +

                    capital_flow

                    +

                    exogenous

                )

                #
                # evolve market
                #

                state = (

                    state

                    +

                    total_deformation

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
