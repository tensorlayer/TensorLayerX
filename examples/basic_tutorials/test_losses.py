import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx

logits = tlx.convert_to_tensor([[0.4, 0.2, 0.8], [1.1, 0.5, 0.3]])
labels = tlx.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

output = tlx.losses.iou_coe(logits, labels, axis=-1)
print(output)

