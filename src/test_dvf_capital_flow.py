from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_capital_flow import DVFCapitalFlow


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

current_state = current_state.unsqueeze(0)

encoder = ObservationEncoder()

capital = DVFCapitalFlow()

latent = encoder(
    current_state
)

output = capital(
    latent
)

print()

print(
    "Encoded shape:",
    latent.shape
)

print(
    "Capital Flow DVF shape:",
    output.shape
)

print()
