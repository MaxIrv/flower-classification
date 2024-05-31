import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2 as trans2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
import pandas as pd


def load_test_data(batch_size):
    transform_normal = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor()
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((250, 250)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(55),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        # Randomly crop to a smaller size and resize back
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  # Random perspective transformation
        # transforms.Normalize(mean = mean, std = std) # Takes each value for the channel, subtracts the mean and divides by the standard deviation (value - mean) / std
    ])

    # Define the transformations
    transformations1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((250, 250))])

    # Load the dataset
    training_dataset = torchvision.datasets.Flowers102(root='./data', split="train",
                                                       download=True, transform=transform_train)
    testing_dataset = torchvision.datasets.Flowers102(root='./data', split="test",
                                                      download=True, transform=transformations1)
    validation_dataset = torchvision.datasets.Flowers102(root='./data', split="val",
                                                         download=True, transform=transformations1)

    # Create the dataloaders
    train_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        testing_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.PReLU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        # self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 102)  # Output layer for 102 classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.fc3(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x


def NetworkAccuracyOnValidation(model, validation_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        # num_class_correct = [0 for i in range(102)]
        # num_class_samples = [0 for i in range(102)]
        total_correct = 0
        total_samples = 0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            # for i in range(len(labels)):
            #     label = labels[i]
            #     pred = predictions[i]
            #     if label == pred:
            #         num_class_correct[label] += 1
            #     num_class_samples[label] += 1

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = 100. * val_correct / val_total

    return val_epoch_loss, val_epoch_acc


def NetworkAccuracyOnTesting(model, device, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    num_class_correct = [0] * 102
    num_class_samples = [0] * 102
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predictions.eq(labels).sum().item()

            c = (predictions == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                num_class_correct[label] += c[i].item()
                num_class_samples[label] += 1

            # for i in range(len(labels)):
            #     label = labels[i]
            #     pred = predictions[i]
            #     if (label == pred):
            #         num_class_correct[label] += 1
            #     num_class_samples[label] += 1

    acc = 100.0 * total_correct / total_samples
    print(f'Accuracy on testing set: {acc} %')

    for i in range(102):
        class_acc = 100.0 * num_class_correct[i] / num_class_samples[i]
        print(f'Accuracy of {i} : {class_acc} %')

    return acc


def NetworkTraining(model, device, train_loader, validation_loader, criterion, optimizer, scheduler, epochs, early_stopping_patience, test_loader):
    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_accuracy_epoch = 0
    best_accuracy = 0
    no_improve_epochs = 0

    for epoch in range(epochs):  # Loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader, 0):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            label_pred = model(images)
            loss = criterion(label_pred, labels)

            # Manually add L2 regularization
            # l2_loss = 0
            # for param in model.parameters():
            #     l2_loss += torch.sum(torch.pow(param, 2))
            # loss += 0.01 * l2_loss  # L2 regularization term

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = label_pred.max(1)
            # predicted = torch.max(label_pred, 1)[1]
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            batch_corr = (predicted == labels).sum()
            batch_acc = batch_corr.item() / len(images)

            # print(f"Epoch Number {epoch}, Index = {i}/{len(train_loader)-1}, Loss = {loss.item()}")

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        val_epoch_loss, val_epoch_acc = NetworkAccuracyOnValidation(model, validation_loader, criterion, device)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

        # scheduler.step(val_epoch_loss)
        scheduler.step()

        if (val_epoch_acc > best_accuracy):
            best_accuracy = val_epoch_acc
            best_accuracy_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best accuracy on validation split: {best_accuracy} at epoch {best_accuracy_epoch}")

    model.load_state_dict(torch.load('best_model.pth'))
    test_acc = NetworkAccuracyOnTesting(test_loader=test_loader, model=model, device=device)
    
    return train_accuracies, train_losses, val_accuracies, val_losses, test_acc

class HyperParameterData:
    def __init__(self, name, values, default):
        self.name = name
        self.values = values
        self.default = default

    def get_values(self):
        return self.values
        
    def get_default(self):
        return self.default


def main(learning_rate, batch_size, weight_decay):
    
    # table_results = pd.DataFrame(columns=["learning_rate", "batch_size", "weight_decay", "train_accuracies", "train_losses", "val_accuracies", "val_losses", "test_accuracy"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 250
    early_stopping_patience = 50

    # Load the data
    train_loader, test_loader, validation_loader = load_test_data(batch_size)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=8)
    
    train_accuracies, train_losses, val_accuracies, val_losses, test_acc = NetworkTraining(
        model, device, train_loader, validation_loader, criterion, optimizer, scheduler, epochs, early_stopping_patience, test_loader
        )
    
    return learning_rate, batch_size, weight_decay, train_accuracies, train_losses, val_accuracies, val_losses, test_acc
    
    
if __name__ == "__main__":
    learning_rate_values = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
    learning_rate_default = 0.0001

    batch_size_values = [8, 16, 32, 64, 128, 256]
    batch_size_default = 64

    weight_decay_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    weight_decay_default = 0.01

    # schedulers = [ lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.3, total_iters=8)]
    # scheduler_default = "LinearLR"

    hyperparameters = [
        HyperParameterData("learning_rate", learning_rate_values, learning_rate_default),
        HyperParameterData("batch_size", batch_size_values, batch_size_default),
        HyperParameterData("weight_decay", weight_decay_values, weight_decay_default),
        # HyperParameterData("scheduler", schedulers, scheduler_default)
    ]
    
    table_results = pd.DataFrame(columns=["learning_rate", "batch_size", "weight_decay", "train_accuracies", "train_losses", "val_accuracies", "val_losses", "test_accuracy"])
    
    for specialParam in hyperparameters:
        for value in specialParam.get_values():
            try:
                
                for hyperparam in hyperparameters:
                    ## Switch on hyperparam.name
                    if hyperparam.name == "learning_rate":
                        learning_rate = hyperparam.default
                    elif hyperparam.name == "batch_size":
                        batch_size = hyperparam.default
                    elif hyperparam.name == "weight_decay":
                        weight_decay = hyperparam.default
                
                if specialParam.name == "learning_rate":
                    learning_rate = value
                elif specialParam.name == "batch_size":
                    batch_size = value
                elif specialParam.name == "weight_decay":
                    weight_decay = value
                    
                print(f"Training for spcial {specialParam.name} = {value}, where learning_rate = {learning_rate}, batch_size = {batch_size}, weight_decay = {weight_decay}")
                learning_rate, batch_size, weight_decay, train_accuracies, train_losses, val_accuracies, val_losses, test_acc = main(learning_rate, batch_size, weight_decay)
                
                table_results.loc[len(table_results.index)] = [
                    learning_rate,
                    batch_size,
                    weight_decay,
                    train_accuracies,
                    train_losses,
                    val_accuracies,
                    val_losses,
                    test_acc
                ]
                
                
                table_results.to_csv("results.csv")
                print(f"Completed for {specialParam.name} = {value}")
            except:
                print(f"Error for {specialParam.name} = {value}")
    
    # main(hyperparameters)