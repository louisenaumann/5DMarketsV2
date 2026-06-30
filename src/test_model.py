from dataset import MarketDataset

from model import FiveDMarkets


dataset = MarketDataset()

sample = dataset[0]

current_state = sample[0]

#
# add batch dimension
#

current_state = current_state.unsqueeze(0)


model = FiveDMarkets()


output = model(

    current_state

)


print()

print(

    "Input shape:",

    current_state.shape

)

print(

    "Final output shape:",

    output.shape

)

print()

print(

    output

)

print()
