import torch
import torch.nn as nn


from layers.observation_encoder import ObservationEncoder

from layers.dvf_correlation import DVFCorrelation
from layers.dvf_sector import DVFSector
from layers.dvf_capital_flow import DVFCapitalFlow
from layers.dvf_exogenous import DVFExogenous

from layers.fusion_operator import FusionOperator
from layers.temporal_memory import TemporalMemory
from layers.future_transition import FutureTransition
from layers.decision_functional import DecisionFunctional
from layers.reconstruction_head import ReconstructionHead


class FiveDMarkets(nn.Module):

    def __init__(self):

        super().__init__()

        #
        # observation encoder
        #

        self.encoder = ObservationEncoder()

        #
        # deformation fields
        #

        self.corr = DVFCorrelation()

        self.sector = DVFSector()

        self.capital = DVFCapitalFlow()

        self.exogenous = DVFExogenous()

        #
        # fusion
        #

        self.fusion = FusionOperator()

        #
        # temporal memory
        #

        self.memory = TemporalMemory()

        #
        # future projection
        #

        self.future = FutureTransition()

        #
        # decision output
        #

        self.reconstruction = ReconstructionHead()
        
        self.decision = DecisionFunctional()

    def forward(self,
                x):

        #
        # encode observations
        #

        latent = self.encoder(
            x
        )

        #
        # deformation fields
        #

        m_a = self.corr(
            latent
        )

        m_b = self.sector(
            latent
        )

        m_c = self.capital(
            latent
        )

        m_d = self.exogenous(
            latent
        )

        #
        # fuse fields
        #

        fused = self.fusion(

            m_a,

            m_b,

            m_c,

            m_d

        )

        #
        # temporary asset collapse
        #

        #fused = torch.mean(

        #    fused,

        #    dim=1

        #)
        fused = fused.view(

            fused.shape[0],

            -1

        )

        #
        # synthetic sequence
        #

        sequence = fused.unsqueeze(1).repeat(

            1,

            5,

            1

        )

        #
        # memory
        #

        memory_output = self.memory(

            sequence

        )

        #
        # future trajectory
        #

        #future_output = self.future(

            #memory_output

        #)

        #
        # decision outputs
        #

        #decision_output = self.decision(

            #future_output

        #)

        #return decision_output
        #return future output

        print("Memory output shape:", memory_output.shape)
                    
        future_output = self.future(

            memory_output

        )

        reconstructed = self.reconstruction(

            future_output

        )

        return reconstructed
