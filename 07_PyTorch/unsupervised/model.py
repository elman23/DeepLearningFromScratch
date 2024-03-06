from typing import Tuple

from torch import nn, Tensor

from layers import ConvLayer, DeconvLayer, DenseLayer
from utils import assert_dim


class PyTorchModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        raise NotImplementedError()


class Autoencoder(PyTorchModel):
    def __init__(self,
                 hidden_dim: int = 28):
        super(Autoencoder, self).__init__()
        self.conv1 = ConvLayer(1, 14, 5, activation=nn.Tanh())
        self.conv2 = ConvLayer(14, 7, 5, activation=nn.Tanh(), flatten=True)

        self.dense1 = DenseLayer(7 * 28 * 28, hidden_dim, activation=nn.Tanh())
        self.dense2 = DenseLayer(hidden_dim, 7 * 28 * 28, activation=nn.Tanh())

        self.conv3 = ConvLayer(7, 14, 5, activation=nn.Tanh())
        self.conv4 = ConvLayer(14, 1, 5, activation=nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 4)

        x = self.conv1(x)
        x = self.conv2(x)
#         import pdb; pdb.set_trace()
        encoding = self.dense1(x)

        x = self.dense2(encoding)

        x = x.view(-1, 7, 28, 28)

        x = self.conv3(x)
        x = self.conv4(x)

        return x, encoding
