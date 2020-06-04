import config
import model
import torch.nn as nn
import torch
from tqdm import tqdm
import utils
import numpy as np

def loss_fn(prediction, target):
    loss_calc = nn.CrossEntropyLoss()
    loss = loss_calc(prediction, target)

    return loss

def train_fn(data_loader, model, device, optimizer, scheduler=None):
    model.train()

    losses = utils.AverageMeter()
    tq0 = tqdm(data_loader, total=len(data_loader))
    for bi, data in tqdm(enumerate(tq0)):
        # load data
        input_ids = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']

        label = data['label']
        sentiment = data['sentiment']

        # prepare input data
        input_ids = input_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)

        label = label.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.long)

        # forward(self, ids, mask, type_ids)
        optimizer.zero_grad()

        out = model(
            ids = input_ids,
            mask = attention_mask,
            type_ids = token_type_ids
        )

        loss = loss_fn(out, sentiment)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))
        tq0.set_postfix(loss_avg=losses.avg)

    return losses.avg




def eval_fn(data_loader, model, device):
    model.eval()

    losses = utils.AverageMeter()

    with torch.no_grad():
        tq0 = tqdm(data_loader, total=len(data_loader))
        for bi, data in tqdm(enumerate(tq0)):
            # load data / ready to input
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']

            label = data['label']
            sentiment = data['sentiment']

            # prepare input data
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)

            label = label.to(device, dtype=torch.long)
            sentiment = sentiment.to(device, dtype=torch.long)

            # forward(self, ids, mask, type_ids)

            out = model(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )

            loss = loss_fn(out, sentiment)
            

            losses.update(loss.item(), input_ids.size(0))
            tq0.set_postfix(loss_avg=losses.avg)

    return losses.avg
