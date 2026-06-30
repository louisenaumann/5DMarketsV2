from dataset import MarketDataset

from layers.observation_encoder import ObservationEncoder

from layers.dvf_correlation import DVFCorrelation


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

#
# add batch dimension
#

current_state = current_state.unsqueeze(0)

encoder = ObservationEncoder()

dvf = DVFCorrelation()

latent = encoder(
    current_state
)

output = dvf(
    latent
)

print()

print(
    "Encoded shape:",
    latent.shape
)

print(
    "DVF output shape:",
    output.shape
)

print()
