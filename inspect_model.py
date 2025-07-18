import torch
state_dict = torch.load("model.pth", map_location="cpu")
for key, value in state_dict.items():
    print(key, value.shape)
