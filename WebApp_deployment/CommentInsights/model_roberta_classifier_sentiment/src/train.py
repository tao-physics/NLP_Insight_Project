import config
import torch
import torch.nn as nn
import pandas as pd

from sklearn import model_selection

from dataset import CommentData
from model import SentimentModel, LabelModel

import transformers
from transformers import RobertaConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import engine

import utils

def run(fold=0):
    # kfold type of data input
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)
    df_train = data[data['kfold'] != fold].reset_index(drop=True)
    df_valid = data[data['kfold'] == fold].reset_index(drop=True)

    train_data = CommentData(
        comments = df_train['Comment'],
        labels = df_train['Label_encoded'],
        sentiments = df_train['Sentiment_encoded']
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size = config.TRAIN_BATCH_SIZE,
        # num_workers = 4
    )

    valid_data = CommentData(
        comments = df_valid['Comment'],
        labels = df_valid['Label_encoded'],
        sentiments = df_valid['Sentiment_encoded']
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_data,
        batch_size = config.VALID_BATCH_SIZE,
        # num_workers = 4
    )

    device = torch.device('cuda')

    model_config = RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    model_config.output_hidden_states = True   

    model = SentimentModel(model_config, config.OUTPUT_SIZE)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # train_fn(data_loader, model, device, optimizer, scheduler=None)
    train_loss_rec = []    
    eval_loss_rec = []

    early_stopping = utils.EarlyStopping(patience=5, mode='min')

    for epoch in range(config.EPOCHS):
        print(f'########### fold = {fold} epoch = {epoch} ############')
        loss_train = engine.train_fn(
            data_loader = train_dataloader,
            model = model,
            device = device,
            optimizer = optimizer,
            scheduler = scheduler
        )

        train_loss_rec.append(loss_train)

        losses_eval = engine.eval_fn(valid_dataloader, model, device)
        eval_loss_rec.append(losses_eval)

        print(f'train_loss = {loss_train}  eval_loss = {losses_eval}')
        # print(f'save model_{fold}.bin')
        # torch.save(model.state_dict(), config.OUTPUT_PATH + f'/model_{fold}.bin')
        early_stopping(losses_eval, model, 
                    model_path = config.OUTPUT_PATH + f'/model_label_{fold}.bin')
        if early_stopping.early_stop:
            print('Early stopping')
            break



if __name__ == '__main__':
    for fold in range(config.TOTAL_FOLDS):
        run(fold)