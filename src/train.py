import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MarketDataset
from model import FiveDMarkets

from config import *


dataset = MarketDataset()

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


model = FiveDMarkets()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

criterion = nn.MSELoss()


loss_history = []


for epoch in range(

        NUM_EPOCHS):

    total_loss = 0

    for current_state, future_true in loader:

        optimizer.zero_grad()

        future_pred = model(

            current_state

        )

        loss = criterion(

            future_pred,

            future_true

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (

        total_loss

        /

        len(loader)

    )

    loss_history.append(
        avg_loss
    )

    print(

        "Epoch:",

        epoch,

        "Loss:",

        avg_loss

    )


print()

print(
    loss_history
)

print()
