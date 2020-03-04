import torch

total += torch.sum(torch.sum(torch.sum(abs(x_b-x_a), 0), 0), 0)/(64.0*3.0*3.0)

total += math.sqrt(torch.sum(torch.sum(pow(x_b-x_a, 2.0), 0), 0), 0)/(64.0*3.0*3.0)



---


