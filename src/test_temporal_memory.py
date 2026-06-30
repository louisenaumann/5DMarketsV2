import torch

from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_correlation import DVFCorrelation
from layers.dvf_sector import DVFSector
from layers.dvf_capital_flow import DVFCapitalFlow
from layers.dvf_exogenous import DVFExogenous

from layers.fusion_operator import FusionOperator

from layers.temporal_memory import TemporalMemory


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

current_state = current_state.unsqueeze(0)


encoder = ObservationEncoder()

corr = DVFCorrelation()
sector = DVFSector()
capital = DVFCapitalFlow()
exo = DVFExogenous()

fusion = FusionOperator()

memory = TemporalMemory()


latent = encoder(
    current_state
)

m_a = corr(latent)
m_b = sector(latent)
m_c = capital(latent)
m_d = exo(latent)

fused = fusion(

    m_a,

    m_b,

    m_c,

    m_d

)

#
# average over assets
#

fused = torch.mean(

    fused,

    dim=1

)

#
# create fake history
#

sequence = fused.unsqueeze(1).repeat(

    1,

    5,

    1

)

#
# small perturbation
#

sequence += torch.randn_like(

    sequence

) * 0.01


output = memory(
    sequence
)

print()

print(
    "Sequence shape:",
    sequence.shape
)

print(
    "Memory output shape:",
    output.shape
)

print()
