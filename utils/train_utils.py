import sys
import torch
import inspect
import numpy as np
import torchmetrics

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

import torchmetrics

class Metrics:
    def __init__(self):
        """
        Class with the different metrics computed by the model.
        """
        self.auroc = torchmetrics.AUROC(task='binary')
        self.accuracy = torchmetrics.Accuracy(task='binary')  # Asegúrate de definirla aquí
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')
        self.reset()  # Llamar a reset para inicializar los estados internos

    def reset(self):
        """
        Reset the metrics.
        """
        self.auroc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def update(self, preds, labels):
        """
        Update the metrics with new predictions and labels.
        """
        self.auroc.update(preds, labels)
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)

    def compute(self):
        """
        Compute and return the metrics.
        """
        return {
            "AUROC": self.auroc.compute().item(),
            "Accuracy": self.accuracy.compute().item(),
            "Precision": self.precision.compute().item(),
            "Recall": self.recall.compute().item(),
            "F1": self.f1.compute().item()
        }

def train_one_epoch(epoch_id,
                    model,
                    loader,
                    optimizer,
                    loss_fn):
    # Tell the model it's going to train
    model.train()
    # Initialize the loss value container and metrics
    loss_epoch = 0
    metrics = Metrics()
    metrics.reset()

# Iterate through the batches in the data loader
    for batch in loader:
        # 1. Zero grad the optimizer
        optimizer.zero_grad()
        # 2. Pass the data to the model
        out = model.forward(batch)
        # Pick the target for loss computation
        label = batch['y'].float()
        # 3. Compute the loss comparing the output of the model to the target
        loss = loss_fn(out, label)
        # 4. Back propagation (compute gradients of the loss with respect to the weights in the model)
        loss.backward()
        # 5. Gradient descent (update the optimizer)
        optimizer.step()

        # Sum loss to get at the end the average loss per epoch
        loss_epoch += loss.item()

        # Compute binary predictions
        preds = torch.sigmoid(out).detach()
        binary_preds = (preds >= 0.5).float()

        # Update metrics
        metrics.update(binary_preds.cpu(), label.cpu())

    # Average the loss for the whole epoch
    loss_epoch = loss_epoch / len(loader)

    # Compute final metrics
    final_metrics = metrics.compute()

    # Print training information
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_  = f"\nTraining Loss: {loss_epoch:.6f}"
    # print(epoch_ + loss_)

    return loss_epoch, final_metrics

def valid_one_epoch(model,
                    loader,
                    loss_fn):
    # Tell the model it's going to evaluate
    model.eval()
    # Initialize the loss value container and metrics
    loss_epoch = 0
    metrics = Metrics()
    metrics.reset()

    # Freeze the gradients (we don't want anymore training)
    with torch.no_grad():
        # Iterate through the batches in the data loader
        for batch in loader:
            # 1. Pass the data through the model
            out = model.forward(batch)
            # Pick the target for loss computation
            label = batch['y'].float()
            # 2. Compute the loss comparing the output of the model to the target
            loss = loss_fn(out, label)

            # Sum loss to get at the end the average loss per epoch
            loss_epoch += loss.item()

            # Compute binary predictions
            preds = torch.sigmoid(out).detach()
            binary_preds = (preds >= 0.5).float()

            # Update metrics
            metrics.update(binary_preds.cpu(), label.cpu())

        # Average the loss for the whole epoch
        loss_epoch = loss_epoch / len(loader)

        # Compute final metrics
        final_metrics = metrics.compute()

        # Print validation information
        loss_  = f"Validation Loss: {loss_epoch:.6f}"
        # print(loss_)

    return loss_epoch, final_metrics

def save_checkpoint(state, filename='checkpioint.pth.tar'):
    torch.save(state, filename)

def get_name_of_scheduler(scheduler):
    """
    Get the name of the scheduler by matching its class name.
    """
    scheduler_name = scheduler.__class__.__name__  # Obtain the name of the class
    if scheduler_name in lr_scheduler.__dict__:   # Verify if the module exists
        return scheduler_name
    return None

def train_net(*,
              nepoch,
              train_dataset,
              valid_dataset,
              train_batch_size,
              valid_batch_size,
              model,
              optimizer,
              criterion,
              scheduler,
              checkpoint_dir,
              tensorboard_dir):
    """
    Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    # Create the Train and Validation DataSets
    loader_train = DataLoader(train_dataset,
                              batch_size = train_batch_size,
                              shuffle    = True,
                              drop_last  = True)
    loader_valid = DataLoader(valid_dataset,
                              batch_size = valid_batch_size,
                              shuffle    = True,
                              drop_last  = True)

    # Initialize loss
    start_loss = np.inf

    # Initialize Tensorflow Writer
    writer = SummaryWriter(tensorboard_dir)

    # Start training
    for i in range(nepoch):
        # Compute training and validation loss nepoch times
        train_loss, train_met = train_one_epoch(i, model, loader_train, optimizer, criterion)
        valid_loss, valid_met = valid_one_epoch(   model, loader_valid,            criterion)

        # Apply scheduler if requested
        if scheduler:
            # ReduceLROnPlateau takes into account validation loss, so we need to evaluate it
            if get_name_of_scheduler(scheduler) == "ReduceLROnPlateau":
                scheduler.step(valid_loss)
            # Other schdulers just reduce LR every some steps, but do not take into account loss
            else:
                scheduler.step()

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': model.state_dict(),
                             'optimizer':  optimizer.state_dict()},
                             f"{checkpoint_dir}/net_checkpoint{i}.pth.tar")
            start_loss = valid_loss

        # Log Losses to TensorBoard
        writer.add_scalar('loss/train', train_loss, i)
        writer.add_scalar('loss/valid', valid_loss, i)

        # Log metrics to TensorBoard
        for metric_name, metric_value in train_met.items():
            writer.add_scalar(f"Metrics/Train/{metric_name}", metric_value, i)
        for metric_name, metric_value in valid_met.items():
            writer.add_scalar(f"Metrics/Valid/{metric_name}", metric_value, i)

        # Log learning rate
        learning_rate = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', learning_rate, i)
        print(f"Epoch {i+1}/{nepoch}, LR: {learning_rate:.6f}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        # print(f"Train Metrics: {train_met}")
        # print(f"Valid Metrics: {valid_met}")

        writer.flush()

    writer.close()

def predict_gen(test_data,
                model,
                batch_size):
    """
    Evaluate model
    """
    # Create Test DataLoader
    loader_test = DataLoader(test_data,
                             batch_size = batch_size,
                             shuffle    = False,
                             drop_last  = False)
    # Tell the model it's going to evaluate
    model.eval()
    # We want no more training
    with torch.autograd.no_grad():
        for batch in loader_test:
            # 1. Pass the data through the model
            out = model.forward(batch)
            # 2. Make the prediction, use Sigmoid for Binary Classification
            y_pred = torch.sigmoid(out)
