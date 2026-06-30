from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_correlation import DVFCorrelation
from layers.dvf_sector import DVFSector
from layers.dvf_capital_flow import DVFCapitalFlow
from layers.dvf_exogenous import DVFExogenous

from layers.fusion_operator import FusionOperator


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


latent = encoder(
    current_state
)

m_a = corr(
    latent
)

m_b = sector(
    latent
)

m_c = capital(
    latent
)

m_d = exo(
    latent
)


output = fusion(

    m_a,

    m_b,

    m_c,

    m_d

)

print()

print(
    "Latent shape:",
    latent.shape
)

print(
    "Fusion output shape:",
    output.shape
)

print()
