import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from tensorlayerx.nn import Module, Parameter, ParameterList, ParameterDict
tlx.set_device(device='CPU', id = 0)

class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params1 = ParameterDict({
                'left': Parameter(tlx.ones((5, 10))),
                'right': Parameter(tlx.zeros((5, 10)))
        })

        self.params2 = ParameterList(
            [Parameter(tlx.ones((10,5))), Parameter(tlx.ones((5,10)))]
        )

    def forward(self, x, choice):
        x = tlx.matmul(x, self.params1[choice])
        x = tlx.matmul(x, self.params2[0])
        x = tlx.matmul(x, self.params2[1])
        return x

input = tlx.nn.Input(shape=(5,5))
net = MyModule()
trainable_weights = net.trainable_weights
print("-----------------------------trainable_weights-------------------------------")
for weight in trainable_weights:
    print(weight)
print("-----------------------------------output------------------------------------")
output = net(input, choice = 'right')
print(output)