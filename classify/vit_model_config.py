
import torch
from torch import nn
import torchvision
from torchvision import transforms
import ast


# pytorch
with open("weights/vit/class_flower_names_v3.txt", "r") as f:
    dict_classes = f.read() 
dict_classes = ast.literal_eval(dict_classes)
state =torch.load('weights/vit/vit_flower_model_v3.pth', map_location=torch.device('cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vit = torchvision.models.vit_b_16(weights=None).to(device)
for parameter in model_vit.parameters():
    parameter.requires_grad = False

model_vit.heads = nn.Linear(in_features=768, out_features=len(dict_classes.keys())).to(device)
model_vit.load_state_dict(state)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit_transforms = pretrained_vit_weights.transforms()