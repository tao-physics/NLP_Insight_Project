import pandas as pd
import config

import torch
import torch.nn as nn
from transformers import BertConfig
from model import SentimentModel, LabelModel
from dataset import CommentData
from tqdm import tqdm
import numpy as np

def predict():
    # kfold type of data input
    data = pd.read_csv(config.TEST_FILE)
    data['Label_encoded'] = 0
    data['Sentiment_encoded'] = 0
    df_test = data

    test_data = CommentData(
        comments = df_test['Comment'],
        labels = df_test['Label_encoded'],
        sentiments = df_test['Sentiment_encoded']
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size = config.TEST_BATCH_SIZE,
        # num_workers = 4
    )

    # model
    device = torch.device('cuda')

    model_config = BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True

    model0 = SentimentModel(model_config, config.OUTPUT_SIZE_SENTIMENT)
    model0.to(device)
    # model0 = nn.DataParallel(model0)
    model0.load_state_dict(torch.load(config.SAVED_MODEL_PATH + '/model_0.bin'))
    model0.eval()

    model1 = SentimentModel(model_config, config.OUTPUT_SIZE_SENTIMENT)
    model1.to(device)
    # model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load(config.SAVED_MODEL_PATH + '/model_1.bin'))
    model1.eval()

    model2 = SentimentModel(model_config, config.OUTPUT_SIZE_SENTIMENT)
    model2.to(device)
    # model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load(config.SAVED_MODEL_PATH + '/model_2.bin'))
    model2.eval() 

    model3 = SentimentModel(model_config, config.OUTPUT_SIZE_SENTIMENT)
    model3.to(device)
    # model3 = nn.DataParallel(model3)
    model3.load_state_dict(torch.load(config.SAVED_MODEL_PATH + '/model_3.bin'))
    model3.eval()

    model4 = SentimentModel(model_config, config.OUTPUT_SIZE_SENTIMENT)
    model4.to(device)
    # model4 = nn.DataParallel(model4)
    model4.load_state_dict(torch.load(config.SAVED_MODEL_PATH + '/model_4.bin'))
    model4.eval()

    # process raw output
    model_prediction = []

    with torch.no_grad():
        tq0 = tqdm(test_dataloader, total=len(test_dataloader))
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

            out0 = model0(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )

            out1 = model1(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )

            out2 = model2(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )

            out3 = model3(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )

            out4 = model4(
                ids = input_ids,
                mask = attention_mask,
                type_ids = token_type_ids
            )


            out = (out0 + out1 + out2 + out3 + out4) / 5
            out = torch.softmax(out, dim=1).cpu().detach().numpy()

            for ix, result in enumerate(out):
                pred = np.argmax(result)
                model_prediction.append(pred)


    sample = pd.read_csv(config.TEST_FILE)
    sample['sentiment_pred'] = model_prediction
    sample.to_csv(config.OUTPUT_PATH + '/pred_sentiment.csv', index=False)



if __name__ == '__main__':
    predict()