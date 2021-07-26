import pandas as pd
from sklearn import model_selection
import config

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_FILE)
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(
        n_splits=config.TOTAL_FOLDS,
        shuffle=False,
        random_state=None
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[config.FOLD_COLS])):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv(config.TRAIN_FOLDS_FILE, index=False)

