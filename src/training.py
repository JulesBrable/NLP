
import time
from tqdm import tqdm
import torch
import os
import logging
import json

from src.utils import get_data, update_json
from src.evaluation import plot_losses, plot_confusion_matrix, predict, make_classif_report


def save_model(model_name, finetuned_model_state_dict, patience, epochs, training_info, es):
    # specify whether there is early stopping or not
    es_suffix = "" if not es else "_es"
    file_name = "model" + es_suffix + ".pth"

    directory = os.path.join("data", "outputs", model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)
    torch.save(finetuned_model_state_dict, file_path)
    with open(
        os.path.join(directory, 'training_summary' + es_suffix + '.json'), 'w'
    ) as f:
        json.dump(training_info, f, indent=4)

    logging.info(f"Model saved successfully to {file_path}")


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device)
        )

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == b_labels).sum().item()
            total_examples += b_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    patience,
    model_name,
    es
):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    start_time = time.time()
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')

        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(avg_train_loss)

        print(f"  Average training loss: {avg_train_loss:.2f}")

        avg_val_loss, accuracy = validate(model, val_dataloader, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Validation Loss: {avg_val_loss:.2f}")
        print(f"Validation Accuracy: {accuracy:.2f}")

        # use early stopping or not
        if es:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_info = {
                    'state_dict': model.state_dict(),
                    'train_time': time.time() - start_time,
                    'epoch': epoch + 1
                }
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")

            if early_stopping_counter >= patience:
                print("Stopping early due to no improvement in validation loss.")
                break
        else:
            best_model_info = {'state_dict': model.state_dict()}

    total_training_time = round(time.time() - start_time, 2)
    print(f"Total training time: {total_training_time:.2f} seconds.")

    training_info = {
        'best_model_epoch': best_model_info.get('epoch', epochs),
        'best_model_train_time': best_model_info.get('train_time', total_training_time),
        'total_training_time': total_training_time,
    }
    save_model(model_name, best_model_info['state_dict'], patience, epochs, training_info, es)
    return train_losses, val_losses, val_accuracies


def run_finetuning(
    tokenizer, model, model_name, params,
    train_dataloader, val_dataloader, test_dataloader,
    es, device
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters to be learned: {n_params}")

    train_losses, val_losses, val_accuracies = train_and_evaluate(
        model, train_dataloader, val_dataloader, optimizer, device, params['epochs'],
        params['patience'], model_name, es
        )

    es_suffix = "_es" if es else ""
    stop = get_data(
        os.path.join("data", "outputs", model_name, 'training_summary_es.json')
        )["best_model_epoch"] if es else None
    _ = plot_losses(
        train_losses, val_losses, stop=stop, save_path=model_name + es_suffix
        )

    start_time = time.time()
    true_labels, predictions = predict(model, test_dataloader, device)
    inference_time = round(time.time() - start_time, 2)
    print("Inference time: ", inference_time)

    update_json(os.path.join(
        "data", "outputs", model_name, "training_summary" + es_suffix + ".json"
        ), 'inference_time', inference_time)

    _ = plot_confusion_matrix(true_labels, predictions, model_name + es_suffix)
    make_classif_report(
        true_labels, predictions,
        os.path.join("data", "outputs", model_name, "classif_report" + es_suffix + ".json")
    )
