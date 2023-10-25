import asyncio
import os
import uuid

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import app.db.crud.users_crud as user_crud
from app.core.neuronal_network.dataset import UserActivityDataset
from app.core.neuronal_network.net import Net
from app.db.database import get_db_ctx



def train_iteration(full_dataset: UserActivityDataset, model: Net, device=torch.device("cpu"), lr: float = 1e-3, batch: int = 4,
                    epochs: int = 500):

    train_size = int(0.7 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.MAPEL()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if os.path.isfile('model.pt'):
        model.load_state_dict(torch.load('model.pt'))

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # print(data, target)
            # print(data)
            output = model(data)
            # print(target)
            # calculate the loss

            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss
            valid_loss += loss.item() * data.size(0)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

    plt.clf()
    plt.ioff()
    plt.plot(list(range(0, epochs)), train_losses, label='train_loss')
    plt.plot(list(range(0, epochs)), valid_losses, label='valid_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(shadow=False, )
    # plt.gca().invert_xaxis()
    plt.show()


async def main():
    net = Net()

    if os.path.isfile('model.pt'):
        net.load_state_dict(torch.load('model.pt'))

    async with get_db_ctx() as db:
        user_db = await user_crud.get_user(uuid.UUID('713d523a-4265-40ec-b42f-3a6474236892'), db)

        dataset = await UserActivityDataset.from_user(user=user_db, db=db)

    train_iteration(dataset, model=net, device=torch.device('cpu'), lr=1e-3, batch=25, epochs=500)
    return

    # dataloader = DataLoader(dataset)

    net.eval()

    x = list(range(len(dataset)))

    y_orig = dataset._y
    print(dataset._x)
    y_pred = net(dataset._x).tolist()

    # print(dataset._x)/
    # print(y_pred)

    plt.plot(x, y_orig, label='Original')
    plt.plot(x, y_pred, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    asyncio.run(main())
