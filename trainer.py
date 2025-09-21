from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from training_configuration import TrainingConfiguration


class Trainer:
    def __init__(self,
                 config: TrainingConfiguration,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 epoch_idx: int):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch_idx = epoch_idx

    def metrics(self,
                output: torch.Tensor,
                target: torch.Tensor,
                batch_losses: list[Any],
                batch_accs: list[Any],
                loss: torch.Tensor,
                data: int,
                batch_idx: int):
        # metrics (no need for softmax)
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            acc = correct / data.size(0)

        batch_losses.append(loss.item())
        batch_accs.append(acc)

        if batch_idx % self.config.log_interval == 0 and batch_idx > 0:
            print(
                'Train Epoch: {} [{}/{}]  Loss: {:.6f}  Acc: {:.6f}'.format(
                    self.epoch_idx,
                    batch_idx * data.size(0),
                    len(self.train_loader.dataset),
                    loss.item(),
                    acc
                )
            )

    def train(self):
        self.model.train()

        batch_losses = []
        batch_accs = []

        # data: tensor batch
        # target: labels
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # move to device
            data = data.to(self.config.device)
            target = target.to(self.config.device)

            # zero grads
            self.optimizer.zero_grad()

            # compute logits for each class
            output = self.model(data)

            # loss on logits (cross_entropy applies log-softmax internally)
            loss = F.cross_entropy(output, target)

            # backward + step
            loss.backward()
            self.optimizer.step()

            self.metrics(output, target, batch_losses, batch_accs, loss, data, batch_idx)
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_accs)
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0
        count_corect_predictions = 0

        # turn off gradient-computation
        with torch.no_grad():
            for data, target in self.test_loader:
                indx_target = target.clone()
                data = data.to(self.config.device)

                target = target.to(self.config.device)

                output = self.model(data)
                # add loss for each mini batch
                test_loss += F.cross_entropy(output, target).item()

                # get probability score using softmax
                prob = F.softmax(output, dim=1)

                # get the index of the max probability
                pred = prob.data.max(dim=1)[1]

                # add correct prediction count
                count_corect_predictions += pred.cpu().eq(indx_target).sum()

            # average over number of mini-batches
            test_loss = test_loss / len(self.test_loader)

            # average over number of dataset
            accuracy = 100. * count_corect_predictions / len(self.test_loader.dataset)

            print(
                '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, count_corect_predictions, len(self.test_loader.dataset), accuracy
                )
            )
        return test_loss, accuracy / 100.0

