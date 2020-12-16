from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

def get_train_datas(data_name : str) -> Bunch:
    data_bunch = fetch_openml(data_name, version=1, data_home=".")
    return data_bunch

def create_model() -> torch.nn.Sequential:
    model = torch.nn.Sequential()
    model.add_module("fc1", torch.nn.Linear(28*28*1, 100))
    model.add_module("reLu1", torch.nn.ReLU())
    model.add_module("fc2", torch.nn.Linear(100, 100))
    model.add_module("reLu2", torch.nn.ReLU())
    model.add_module("fc3", torch.nn.Linear(100, 10))
    return model

def train(model : torch.nn.Sequential, epoch : int, data_loader : DataLoader, optimizer, loss_func) -> None:
    model.train()
    for data, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_func(outputs, targets)

        loss.backward()
        optimizer.step()
    print("Epoch : {} is ended.".format(epoch))

def test(model : torch.nn.Sequential, data_loader) -> None:
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()
    
    data_num = len(data_loader.dataset)
    print("acc : {} / {} ({:.2f}%)".format(correct, data_num, 100.0 * correct / data_num))

def main():
    mnist = get_train_datas("mnist_784")
    x = mnist.data.astype(np.float32) / 255
    y = mnist.target.astype(np.int32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    data_set_train = TensorDataset(x_train, y_train)
    data_set_test = TensorDataset(x_test, y_test)

    model = create_model()

    loader_train = DataLoader(data_set_train, batch_size=64, shuffle=True)
    loader_test = DataLoader(data_set_test, batch_size=64, shuffle=False)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    for epoch in range(3):
        train(model, epoch, loader_train, optimizer, loss_func)
    test(model, loader_test)

if __name__ == "__main__":
    main()  