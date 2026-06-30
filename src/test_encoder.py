import torch

from dataset import MarketDataset
from layers.observation_encoder import ObservationEncoder


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

#
# add batch dimension
#

current_state = current_state.unsqueeze(0)

model = ObservationEncoder()

output = model(
    current_state
)

print()

print(
    "Input shape:",
    current_state.shape
)

print(
    "Output shape:",
    output.shape
)

print()
