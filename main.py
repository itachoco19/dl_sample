from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_train_datas(data_name : str) -> Bunch:
    data_bunch = fetch_openml(data_name, version=1, data_home=".")
    return data_bunch

def main():
    mnist = get_train_datas("mnist_784")
    x = mnist.data / 255
    y = [int(i) for i in mnist.target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    data_set_train = TensorDataset(x_train, y_train)
    data_set_test = TensorDataset(x_test, y_test)

    loder_train = DataLoader(data_set_train, batch_size=64, shuffle=True)
    loder_test = DataLoader(data_set_test, batch_size=64, shuffle=False)

    model = torch.nn.Sequential()
    model.add_module("fc1", torch.nn.Linear(28*28*1, 100))
    model.add_module("reLu1", torch.nn.ReLU())
    model.add_module("fc2", torch.nn.Linear(100, 100))
    model.add_module("reLu2", torch.nn.ReLU())
    model.add_module("fc3", torch.nn.Linear(100, 10))
    print(model)

if __name__ == "__main__":
    main()  