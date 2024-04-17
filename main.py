import numpy as np
import torch

from src.utils import get_params, get_data, get_args
from src.preprocessing import clean_data, get_features, split_data, get_dataloader
from src.training import run_finetuning
from conf.config import models

args = get_args()
es = args.es

raw_tokens = get_data("data/tokens.yml")
raw_data = get_data("data/entities.json")
params = get_params('conf/params.yaml')

torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokens = {key: value['start'] for key, value in raw_tokens.items()}
data = clean_data(raw_data, tokens)
X, y = get_features(data)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, params)

for model_name, classes in models.items():
    tokenizer_class, model_class = classes
    tokenizer = tokenizer_class.from_pretrained(model_name)

    train_dataloader, val_dataloader, test_dataloader = [
        get_dataloader(
            features, labels, tokenizer, batch_size=params["batch_size"]
        ) for features, labels in zip([X_train, X_val, X_test], [y_train, y_val, y_test])
    ]

    model = model_class.from_pretrained(
        model_name, num_labels=len(np.unique(y_train))
        )
    model.to(device)
    model_name = model_name.split('/')[-1]
    run_finetuning(
        tokenizer, model, model_name, params,
        train_dataloader, val_dataloader, test_dataloader,
        es=True, device=device
    )
