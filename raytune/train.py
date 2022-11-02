import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from dataset import load_dataset
import dataset
from dataset import ImageDataset
from model import DenseNet121
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_chexpert(ray_config, checkpoint_dir=None, data_dir=None):
    net = DenseNet121(num_classes=3).to(device)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=ray_config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_df, valid_df = dataset.assign_label(cfg.train_csv, cfg.valid_csv)
    
    train_datset, val_dataset = load_dataset(train_df, valid_df)
    train_loader = DataLoader(train_dataset,
                          batch_size=int(ray_config["batch_size"]),
                          num_workers=cfg.num_workers,
                          shuffle=True,
                          drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=int(ray_config["batch_size"]),
                            num_workers=cfg.num_workers,
                            shuffle=False,
                            drop_last=False)

    for epoch in range(1, 2):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")