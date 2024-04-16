import requests
import os
import torch


def download_finetuned_model(model_name, remote, local_folder, es):
    es = '_es' if es else ""
    # make request to the S3 bucket
    url = f'{remote}/{model_name}/model{es}.pth'
    response = requests.get(url)
    response.raise_for_status()
    # saved finetuned model locally
    save_directory = os.path.join(local_folder, model_name)
    os.makedirs(save_directory, exist_ok=True)
    file_path = os.path.join(save_directory, f'model{es}.pth')

    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path


def init_model(model_name, models, num_labels):
    tokenizer_class, model_class = models[model_name]
    model = model_class.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    return model, tokenizer


def get_finetuned_model(
    model_name,
    models,
    device,
    es: bool = False,
    remote="https://minio.lab.sspcloud.fr/jbrablx/nlp/data/outputs",
    local_folder='data',
    num_labels: int = 2
):
    model, tokenizer = init_model(model_name, models, num_labels=num_labels)
    model_name = model_name.split('/')[-1]
    file_path = download_finetuned_model(model_name, remote, local_folder, es)
    model_state_dict = torch.load(file_path, map_location=torch.device(device))
    model.load_state_dict(model_state_dict)
    return model, tokenizer


def predict_from_finetuned_model(model, tokenizer, inputs):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    predicted_classes = probabilities.argmax(dim=-1)
    return predicted_classes.tolist(), probabilities
