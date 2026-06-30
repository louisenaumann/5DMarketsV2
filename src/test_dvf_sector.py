from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_sector import DVFSector


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

current_state = current_state.unsqueeze(0)

encoder = ObservationEncoder()

sector = DVFSector()

latent = encoder(
    current_state
)

output = sector(
    latent
)

print()

print(
    "Encoded shape:",
    latent.shape
)

print(
    "Sector DVF shape:",
    output.shape
)

print()
