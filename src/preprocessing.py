import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def clean_data(raw_data, tokens):

    # flatten the raw data and split each row based on '\n' symbol
    data = [elem for d in raw_data.values() for elem in d.split("\n")]

    # remove token symbols and create a df
    # where each column is a feature (based on the tokens dictionary)
    rows = []
    for obs in data:
        row = {}
        parts = obs.split()
        for part in parts:
            for key, tag in tokens.items():
                if part.startswith(tag):
                    row[key] = part.replace(tag, '')
                    break
        rows.append(row)
    df = pd.DataFrame(rows, columns=tokens.keys())

    # drop missing values
    df = df.dropna(how='all').reset_index(drop=True)
    return df


def get_features(data, target_name: str = "surname_household"):
    y = (~data[target_name].isna()).astype(int).values
    X_features = data.drop(columns=[target_name])
    X = (
        X_features
        .fillna('')
        .apply(lambda row: ' '.join(row[row != '']), axis=1)
        .to_numpy(dtype=str)
    )
    return X, y


def split_data(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], stratify=y, random_state=params['seed']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=params['test_size'], stratify=y_train, random_state=params['seed']
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataloader(X_sample, y_sample, tokenizer, batch_size: int):
    """
    Create a DataLoader for training or evaluation,
    wrapping the data and the tokenizer with specified batch size.
    """
    X_tokenized = tokenizer(
        list(X_sample),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
        add_special_tokens=True
        )
    y_sample = torch.tensor(y_sample, dtype=torch.int64)
    dataset = TensorDataset(X_tokenized['input_ids'], X_tokenized['attention_mask'], y_sample)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
