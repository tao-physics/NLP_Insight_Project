# the basic idea
# use bert to encode the sentence
# use encoded features to generate the classes

import torch
import torch.nn as nn
import transformers
from transformers import BertModel

import config

class SentimentModel(transformers.BertPreTrainedModel):
    def __init__(self, conf, output_size):
        super(SentimentModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.linear = nn.Linear(768, output_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, ids, mask, type_ids):
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        # pooler_output (torch.FloatTensor: of shape (batch_size, hidden_size))
        # hidden_states (tuple(torch.FloatTensor), optional, returned when config.output_hidden_states=True)
        sequence_output, pooled_output, hiddens = self.bert(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = type_ids
        )

        x = sequence_output[:,0,:]
        # sequence_output.mean(dim=1)
        x = self.drop(x)
        out = self.linear(x)

        return out

class LabelModel(transformers.BertPreTrainedModel):
    def __init__(self, conf, output_size):
        super(LabelModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.linear = nn.Linear(768, output_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, ids, mask, type_ids):
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        # pooler_output (torch.FloatTensor: of shape (batch_size, hidden_size))
        # hidden_states (tuple(torch.FloatTensor), optional, returned when config.output_hidden_states=True)
        sequence_output, pooled_output, hiddens = self.bert(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = type_ids
        )

        x = sequence_output[:,0,:]
        # sequence_output.mean(dim=1)
        x = self.drop(x)
        out = self.linear(x)

        return out

        