import matplotlib.pyplot as plt
import torch
import sklearn.metrics as m
from typing import Union
import os
import json

from src.utils import update_json


def plot_losses(
    train_losses,
    val_losses,
    stop: Union[int, None] = None,
    save_path: Union[str, None] = None,
    save_prefix: str = os.path.join("assets", "plots", "losses")
):
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    fig, ax = plt.subplots(figsize=(10, 5))
    if stop:
        ax.axvline(stop, linestyle='--', color='r', label='Early stopping')
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_title(f'Losses for {save_path.split("/")[-1]}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(save_prefix, save_path) + '.png')
    return fig


def predict(model, dataloader, device):
    model.eval()
    probabilities, predictions, true_labels = [], [], []

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
        )

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            probas = torch.softmax(logits, dim=-1)

        preds = torch.argmax(logits, dim=1)
        probabilities.extend(probas[:, 1].cpu().numpy())
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())
    return true_labels, predictions, probabilities


def plot_confusion_matrix(
    true_labels,
    predictions,
    save_path: Union[str, None] = None,
    save_prefix: str = os.path.join("assets", "plots", "confusion"),
):
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    cm = m.confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = m.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix for {save_path.split("/")[-1]}')

    if save_path:
        plt.savefig(os.path.join(save_prefix, save_path) + '.png')
    return fig


def make_classif_report(y_true, y_pred, filepath):
    report = m.classification_report(y_true, y_pred, output_dict=True)
    with open(filepath, 'w') as file:
        json.dump(report, file, indent=4)


def make_full_report(
    true_labels, predictions, probabilities,
    model_name, inference_time, es_suffix
):
    metrics_dict = {
        'inference_time': inference_time,
        'test_accuracy': m.accuracy_score(true_labels, predictions),
        'test_f1': m.f1_score(true_labels, predictions),
        'test_roc_auc': m.roc_auc_score(true_labels, probabilities)
    }

    for k, v in metrics_dict.items():
        update_json(os.path.join(
            "data", "outputs", model_name, "training_summary" + es_suffix + ".json"
            ), k, v)

    make_classif_report(
        true_labels, predictions,
        os.path.join("data", "outputs", model_name, "classif_report" + es_suffix + ".json")
    )
