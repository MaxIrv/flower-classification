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
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    # Define the transformations
    transformations1 = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((250, 250))])

    # Load the dataset
    training_dataset = torchvision.datasets.Flowers102(root='../data', split="train",
                                                       download=True, transform=transform_train)
    testing_dataset = torchvision.datasets.Flowers102(root='../data', split="test",
                                                      download=True, transform=transformations1)
    validation_dataset = torchvision.datasets.Flowers102(root='../data', split="val",
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
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(256 * 3 * 3, 512)
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
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def NetworkAccuracyOnValidation(model, validation_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
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

    acc = 100.0 * total_correct / total_samples
    print(f'Accuracy on testing set: {acc} %')

    for i in range(102):
        class_acc = 100.0 * num_class_correct[i] / num_class_samples[i]
        print(f'Accuracy of {i} : {class_acc} %')

    return acc


def NetworkTraining(model, device, train_loader, validation_loader, criterion, optimizer, scheduler, epochs, early_stopping_patience, test_loader, scheduler_name):
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

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = label_pred.max(1)
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

        if (scheduler_name == "ReduceLROnPlateau"):
            scheduler.step(val_epoch_loss)
        else:
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


def main(learning_rate, batch_size, weight_decay, optimizer_name, scheduler_name):
                        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 250
    early_stopping_patience = 50

    # Load the data
    train_loader, test_loader, validation_loader = load_test_data(batch_size)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_name == "LinearLR":
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=0.3, total_iters=10)
    
    train_accuracies, train_losses, val_accuracies, val_losses, test_acc = NetworkTraining(
        model, device, train_loader, validation_loader, criterion, optimizer, scheduler, epochs, early_stopping_patience, test_loader, scheduler_name
        )
    
    return learning_rate, batch_size, weight_decay, optimizer_name, scheduler_name, train_accuracies, train_losses, val_accuracies, val_losses, test_acc
            
if __name__ == "__main__":
    learning_rate_default = 0.0005

    batch_size_default = 8

    weight_decay_default = 0.01
    
    optimizers = ["Adam", "AdamW", "RMSprop"]
    schedulers = ["ReduceLROnPlateau", "CosineAnnealingLR", "LinearLR"]
    
    table_results = pd.DataFrame(columns=[
        "learning_rate", "batch_size", "weight_decay", "optimizer", "scheduler", "train_accuracies", "train_losses", "val_accuracies", "val_losses", "test_accuracy"])
    
    for optimizer_name in optimizers:
        for scheduler_name in schedulers:
            try:
                print(f"Starting for {optimizer_name} = {scheduler_name}")
                learning_rate, batch_size, weight_decay, optimizer_name, scheduler_name, train_accuracies, train_losses, val_accuracies, val_losses, test_acc = main(learning_rate_default, batch_size_default, weight_decay_default, optimizer_name, scheduler_name)
                
                table_results.loc[len(table_results.index)] = [
                    learning_rate,
                    batch_size,
                    weight_decay,
                    optimizer_name,
                    scheduler_name,
                    train_accuracies,
                    train_losses,
                    val_accuracies, 
                    val_losses,
                    test_acc
                ]
                
                table_results.to_csv("sched_optim_test_results.csv")
                print(f"Completed for {optimizer_name} = {scheduler_name}")
            except:
                print(f"Failed for {optimizer_name} = {scheduler_name}")
    
    