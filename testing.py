import numpy as np
import torch


se3_cur = np.eye(4)

se3_test = [se3_cur,se3_cur,se3_cur]

se3_tensor = torch.tensor(se3_test)

print(se3_tensor.shape)
