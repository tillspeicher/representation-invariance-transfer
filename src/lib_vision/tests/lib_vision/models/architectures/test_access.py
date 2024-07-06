from torch import nn

from lib_vision.models.architectures import access


def test_one_element_access():
    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin_layer = nn.Linear(100, 10)

        def forward(self, x):
            return self.lin_layer(x)

    net = Network()
    assert access.get_layer(net, "lin_layer") is net.lin_layer


def test_chained_access():
    class InnerNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(100, 100)
            self.layer2 = nn.Linear(100, 100)

        def forward(self, x):
            x = self.layer1(x)
            return self.layer2(x)

    class OuterNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = InnerNetwork()
            self.linear = nn.Linear(100, 10)

        def forward(self, x):
            x = self.layers(x)
            return self.linear(x)

    net = OuterNetwork()
    assert access.get_layer(net, "layers.layer1") is net.layers.layer1
    assert access.get_layer(net, "layers.layer2") is net.layers.layer2
    assert access.get_layer(net, "layers") is net.layers


def test_index_access():
    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 100),
                nn.Linear(100, 10),
            )

        def forward(self, x):
            return self.layers(x)

    net = Network()
    assert access.get_layer(net, "layers[0]") is net.layers[0]
    assert access.get_layer(net, "layers[1]") is net.layers[1]
