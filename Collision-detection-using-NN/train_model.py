from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    losses = []
    train_loss = []
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer, gamma= 0.999)
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        training_loss = 0
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            output = model(sample['input'])
            output = torch.reshape(output, (-1,))
            loss = loss_function(output, sample['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(test_loss)
        train_loss.append(training_loss)

    file = "saved/saved_model.pkl"
    torch.save(model.state_dict(), file, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    no_epochs = 25
    train_model(no_epochs)
