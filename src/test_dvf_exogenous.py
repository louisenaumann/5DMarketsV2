from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_exogenous import DVFExogenous


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

current_state = current_state.unsqueeze(0)

encoder = ObservationEncoder()

exo = DVFExogenous()

latent = encoder(
    current_state
)

output = exo(
    latent
)

print()

print(
    "Encoded shape:",
    latent.shape
)

print(
    "Exogenous DVF shape:",
    output.shape
)

print()
