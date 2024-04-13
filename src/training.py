
import time
from tqdm import tqdm
import torch


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


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, device, epochs):
    train_losses = []
    val_losses = []
    val_accuracies = []
    start = time.time()

    for epoch_i in range(epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')

        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(avg_train_loss)

        print(f"  Average training loss: {avg_train_loss:.2f}")

        avg_val_loss, accuracy = validate(model, val_dataloader, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Validation Loss: {avg_val_loss:.2f}")
        print(f"Validation Accuracy: {accuracy:.2f}")

    print(f"Training time (seconds): {round(time.time() - start, 2)}")
    return train_losses, val_losses, val_accuracies
