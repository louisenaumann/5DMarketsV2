from dataset import MarketDataset

from model import FiveDMarkets

from layers.reconstruction_head import ReconstructionHead


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

current_state = current_state.unsqueeze(0)


model = FiveDMarkets()

reconstruction = ReconstructionHead()


#
# IMPORTANT:
# temporarily model still outputs decision head
# so we test manually
#

latent_output = model.encoder(
    current_state
)

m_a = model.corr(latent_output)

m_b = model.sector(latent_output)

m_c = model.capital(latent_output)

m_d = model.exogenous(latent_output)

fused = model.fusion(

    m_a,

    m_b,

    m_c,

    m_d

)

fused = fused.mean(
    dim=1
)

sequence = fused.unsqueeze(1).repeat(

    1,

    5,

    1

)

memory_output = model.memory(
    sequence
)

future_output = model.future(
    memory_output
)

reconstructed = reconstruction(
    future_output
)

print()

print(
    "Future latent shape:",
    future_output.shape
)

print(
    "Reconstructed shape:",
    reconstructed.shape
)

print()
