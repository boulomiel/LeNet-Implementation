import os
from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time

from lenet5 import LeNet5
from system_configuration import SystemConfiguration
from trainer import Trainer
from training_configuration import TrainingConfiguration


def setup_system(system_config: SystemConfiguration):
    torch.manual_seed(system_config.seed)
    if torch.backends.mps.is_available():
        torch.backends.cudnn.benchmark.enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def start():
    setup_system(SystemConfiguration())
    training_configuration = TrainingConfiguration()

    batch_size_to_set = training_configuration.batch_size
    num_workers_to_set = training_configuration.num_workers
    epoch_num_to_set = training_configuration.epochs_count

    # if GPU is available use training config,
    # else lower batch_size, num_workers and epochs count
    if torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 5

    lenet5_model = LeNet5()
    lenet5_model.to(device)

    (train_loader, test_loader) = lenet5_model.get_data(batch_size=batch_size_to_set,
                                                        data_root=training_configuration.data_root,
                                                        num_workers=num_workers_to_set)

    trainer = Trainer(
        model=lenet5_model,
        config=training_configuration,
        optimizer=torch.optim.SGD(
            lenet5_model.parameters(),
            lr=training_configuration.learning_rate,
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        epoch_idx=epoch_num_to_set,
    )

    best_loss = torch.tensor(np.inf)

    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])

    # epoch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])

    # training time measurement
    t_begin = time.time()
    for epoch in range(training_configuration.epochs_count):

        train_loss, train_acc = trainer.train()

        epoch_train_loss = np.append(epoch_train_loss, [train_loss])

        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs_count - elapsed_time

        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )

        if epoch % training_configuration.test_interval == 0:
            current_loss, current_accuracy = trainer.validate()

            epoch_test_loss = np.append(epoch_test_loss, [current_loss])

            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])

            if current_loss < best_loss:
                best_loss = current_loss

    print("Total time: {:.2f}, Best Loss: {:.3f}".format(time.time() - t_begin, best_loss))
    return (lenet5_model,
            epoch_train_loss,
            epoch_train_acc,
            epoch_test_loss,
            epoch_test_acc)


def save(path: str, generated_model: nn.Module) -> str:
    # make sure to transfer model to cpu
    generated_model.to("cpu")
    # save state_dict
    torch.save(generated_model.state_dict(), path)
    return model_path


def load(path: str) -> nn.Module:
    lenet5_model = LeNet5()
    lenet5_model.load_state_dict(torch.load(path))
    return lenet5_model

def prediction(model: nn.Module,
               training_config: TrainingConfiguration,
               batch_input) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred_indices, pred_probs) as NumPy arrays of shape [B].
    Assumes model outputs logits of shape [B, C].
    """
    # turn off gradient computation
    with torch.no_grad():
        # send model to gpu
        model.to(training_config.device)

        # it is important to do model.eval() before predication
        # so the model is not in training behavior
        model.eval()
        data = batch_input.to(training_config.device)
        # output is typically logits of shape [B, C] (batch size × number of classes).
        output = model(data)

        # get probability score using softmax
        prob = F.softmax(output, dim=1)

        # get the max probability
        pred_prob, pred_index = prob.data.max(dim=1)

    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()

def run_inference(model: nn.Module,
                   training_config: TrainingConfiguration):
    batch_size = 5
    model.to(training_config.device)
    test = DataLoader(
        datasets.MNIST(
            root=training_config.data_root,
            train=False,
            download=True,
            transform= transforms.functional.to_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = DataLoader(
        datasets.MNIST(
            root=training_config.data_root,
            train=False,
            download=True,
            transform= image_transform,
        ),
        shuffle=False,
        num_workers=1
    )
    pred: np.ndarray = np.array([])
    prob: np.ndarray = np.array([])

    # pass the loaded model
    for data, label in test_transform:
        i, p = prediction(model, training_config, data)
        pred = np.append(pred, i)
        prob = np.append(prob,p)

    plt.rcParams["figure.figsize"] = (3, 3)
    for images, _ in test:
        for i, img in enumerate(images):
            img = transforms.functional.to_pil_image(img)
            plt.imshow(img, cmap='gray')
            plt.gca().set_title('Prediction: {0}, Prob: {1:.2}'.format(pred[i], prob[i]))
            plt.show()
        break


if __name__ == '__main__':
    models = "models"
    model_file_name = "lenet5_mnist.pt"
    if not os.path.exists(models):
        os.makedirs(models)
    model_path = os.path.join(models, model_file_name)
    # model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = start()
    # save(path=model_path, model=model)
    model = load(path=model_path)
    run_inference(model=model, training_config=TrainingConfiguration())
