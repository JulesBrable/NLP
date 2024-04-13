import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def clean_data(raw_data, tokens):

    data = [elem for d in raw_data.values() for elem in d.split("\n")]
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

    return pd.DataFrame(rows, columns=tokens.keys())


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


def get_dataloader(X_sample, y_sample, tokenizer, batch_size: int):
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