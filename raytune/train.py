import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from config import cfg
from models import select_model
from dataset import get_dataloader

import wandb

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import datetime as dt

def train(model, optimizer, criterion, train_loader, device):
    model.to(device)
    model.train()
    train_loss = []
    for data, label in enumerate(train_loader):
    #for data, label in tqdm(iter(train_loader)):
        #data, label = data.float().to(device), label.long().to(device)
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # if scheduler is not None:
    #     scheduler.step()
    return np.mean(train_loss)


def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)

            model_pred = model(data)
            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    return np.mean(val_loss), accuracy_score(true_labels, model_preds)


def save_checkpoint(args, model, device, mode="best"):
    rand_input = torch.rand(2, 1, args.img_size, args.img_size, args.img_size).cuda(device)
    traced_net = torch.jit.trace(model.module, rand_input)

    # wandb 사용 여부
    if args.sweep or args.wandb:
        os.makedirs(f"logs/{wandb.run.name}", exist_ok=True)
        filename = f"{wandb.run.name}/{mode}_model_jit.pt"
    else:
        logname = dt.datetime.now().strftime("%y-%m-%d-%h")
        os.makedirs(f"logs/{logname}", exist_ok=True)
        filename = f"{logname}/{mode}_model_jit.pt"

    filename = os.path.join("logs", filename)
    traced_net.save(filename)
    print("Saving checkpoint", filename)


def train_chexpert(args, ray_config, model, device):
    best_score = 0
    train_loader, val_loader = get_dataloader(cfg.data_dir)
#    optimizer = torch.optim.SGD(model.parameters(), lr=ray_config["lr"], momentum=0.9)
    optimizer = torch.optim.Adam(
        model.parameters(),
        #lr=ray_config["lr"],
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, cfg.epochs + 1):
        #model, optimizer, criterion, train_loader, device)
        train_loss = train(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
        )
        print(f"[Train] Epoch {epoch:03} | Loss {train_loss:.3f}")
        if args.wandb or args.sweep:
            wandb.log({"train_loss": train_loss}, step=epoch)

        if epoch % args.val_every == 0:
            val_loss, val_acc = validation(
                model,
                criterion,
                val_loader,
                device,
            )
            print(f"[Valid] Epoch {epoch:03} | Loss {val_loss:.3f} | ACC {val_acc:.4f}")

            if args.wandb or args.sweep:
                wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            if best_score < val_acc:
                best_score = val_acc
                save_checkpoint(args=args, model=model, device=device)


if __name__ == "__main__":
    train_chexpert(cfg)
