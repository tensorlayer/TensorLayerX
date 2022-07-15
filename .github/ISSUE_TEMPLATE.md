### New Issue Checklist

- [ ] I have read the [Contribution Guidelines](https://github.com/tensorlayer/tensorlayer/blob/master/CONTRIBUTING.md)
- [ ] I searched for [existing GitHub issues](https://github.com/tensorlayer/tensorlayer/issues)

### Issue Description

[INSERT DESCRIPTION OF THE PROBLEM]

### Reproducible Code

- Which OS are you using ?
- Please provide a reproducible code of your issue. Without any reproducible code, you will probably not receive any help.

[INSERT CODE HERE]

```python
# ======================================================== #
###### THIS CODE IS AN EXAMPLE, REPLACE WITH YOUR OWN ######
# ======================================================== #

import tensorflow as tf
import tensorlayerx as tl

net_in = tl.layers.Input((3, 64))

net = tl.nn.Linear(out_features=25, in_features=64, act='relu')

print("Output Shape:", net(net_in).shape) ### Output Shape: [None, 25]

# ======================================================== #
###### THIS CODE IS AN EXAMPLE, REPLACE WITH YOUR OWN ######
# ======================================================== #
```



