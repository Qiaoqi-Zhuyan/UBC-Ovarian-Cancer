import math
import torch
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score


def train_eval(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_data_loader: Iterable,
        val_data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        loss_scaler,
        mixup_fn=Mixup,
        lr_schedule_values=None,
        use_amp=False
):
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0

        model.train()
        with tqdm(train_data_loader, desc='Train') as t:
            for x, y in t:
                t.set_description(f"Epoch [{epoch} / {epoch_num}]")
                x = x.to('cuda', non_blocking=True)
                y = y.to('cuda', non_blocking=True)

                if mixup_fn is not None:
                    x, y = mixup_fn(x, y)

                if use_amp:
                    with torch.amp.autocast:
                        y_pred = model(x)
                else:
                    y_pred = model(x)

                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_ = torch.argmax(y_pred, dim=1)
                    #train_correct += (y_ == y).sum().item()
                    #total = train_total
                    #total += y.size(0)
                    #training_acc = train_correct / train_total

                    training_acc = accuracy(y_, y)
                    training_loss = loss.item()
                    y = y.to('cpu')
                    y_ = y_.to('cpu')
                    training_balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, y_))

                t.set_postfix(loss=training_loss, training_acc=training_acc, training_BA=training_balanced_accuracy)
                logger.info(f'Train:'
                            f'Epoch: [{epoch} / {epoch_num}, '
                            f'Training Loss: {training_loss}, '
                            f'Training Accuracy: {training_acc},'
                            f'Training Balanced Accuracy: {training_balanced_accuracy}')
                torch.save(model, save_model_name)

            val_correct = 0
            val_total = 0

            model.eval()
            with torch.no_grad():
                with tqdm(val_dataloader, desc="Val") as t2:
                    for x, y in t2:
                        x = x.to('cuda', non_blocking=True)
                        y = y.to('cuda', non_blocking=True)
                        y_p = model(x)
                        # _, pred = torch.max(y_p, 1)
                        pred = torch.argmax(y_p, dim=1)

                        val_total += y.size(0)
                        val_correct += (pred == y).sum().item()
                        val_acc = '{:.4f}'.format(val_correct / val_total)

                        y = y.to('cpu')
                        pred = pred.to('cpu')
                        balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

                        t2.set_postfix(val_acc=val_acc, val_BA=balanced_accuracy)

                        logger.info(f'Val:'
                                    f'Epoch: [{epoch} / {epoch_num}, '
                                    f'Val Accuracy: {val_acc},'
                                    f'val Balanced Accuracy: {balanced_accuracy}')


def origin_data_test(origin_dataset_loader, model):

    model.eval()
    with tqdm(origin_dataset_loader, desc="test") as t:
        for x, y in t:
            x = x.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            y_p = model(x)
            # _, pred = torch.max(y_p, 1)
            pred = torch.argmax(y_p, dim=1)

            val_total += y.size(0)
            val_correct += (pred == y).sum().item()
            val_acc = '{:.4f}'.format(val_correct / val_total)

            y = y.to('cpu')
            pred = pred.to('cpu')
            balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

            t2.set_postfix(val_acc=val_acc, val_BA=balanced_accuracy)

            logger.info(f'Val:'
                        f'Epoch: [{epoch} / {epoch_num}, '
                        f'Val Accuracy: {val_acc},'
                        f'val Balanced Accuracy: {balanced_accuracy}')



if __name__ == "__main__":
    pass
