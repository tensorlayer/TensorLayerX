import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.linear3 = Linear(out_features=10, act=tlx.ReLU, in_features=800)

    def forward(self, x, foo=None):
        z = self.linear1(x)
        z = self.linear2(z)
        out = self.linear3(z)
        # if foo is not None:
        #     out = tlx.relu(out)
        return out

model = CustomModel()

layer_node = model.node_build(tlx.nn.Input(shape=(3, 784)))
for node in layer_node:
    print(node.node_index)
    print(node.name)