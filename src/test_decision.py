import torch

from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_correlation import DVFCorrelation
from layers.dvf_sector import DVFSector
from layers.dvf_capital_flow import DVFCapitalFlow
from layers.dvf_exogenous import DVFExogenous

from layers.fusion_operator import FusionOperator
from layers.temporal_memory import TemporalMemory
from layers.future_transition import FutureTransition
from layers.decision_functional import DecisionFunctional


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

future = FutureTransition()

decision = DecisionFunctional()


latent = encoder(current_state)

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

fused = torch.mean(
    fused,
    dim=1
)

sequence = fused.unsqueeze(1).repeat(
    1,
    5,
    1
)

sequence += torch.randn_like(
    sequence
) * 0.01

memory_output = memory(
    sequence
)

future_output = future(
    memory_output
)

decision_output = decision(
    future_output
)

print()

print(
    "Future shape:",
    future_output.shape
)

print(
    "Decision shape:",
    decision_output.shape
)

print()

print(
    decision_output
)

print()
