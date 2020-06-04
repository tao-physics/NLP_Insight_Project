import config
import pandas as pd
import torch

class CommentData():
    def __init__(self, comments, labels, sentiments):
        self.comments = comments
        self.labels = labels
        self.sentiments = sentiments

        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = ' '.join(self.comments[item].lower().split())
        label = self.labels[item]
        sentiment = self.sentiments[item]

        # model input

        comment_tk = self.tokenizer.encode_plus(comment, max_length=config.MAX_LEN, pad_to_max_length=True)
        input_ids = comment_tk['input_ids']
        token_type_ids = comment_tk['token_type_ids']
        attention_mask = comment_tk['attention_mask']

        data = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }

        return data
        

# dataset test

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE)
    dset = CommentData(
        comments = df['Comment'],
        labels = df['Label_encoded'],
        sentiments = df['Sentiment_encoded']
    )

    print(dset[99])