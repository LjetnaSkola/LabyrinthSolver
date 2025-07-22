import torch
from torchvision import models
import sys
def conversion():
    # Broj klasa i ulazni model
    num_classes = 6
    model_path = "model.pth"
    
    # Inicijalizacija modela
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Dummy ulaz
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Eksport u ONNX
    torch.onnx.export(model, dummy_input, "./rpi/model.onnx", input_names=["input"], output_names=["output"])
    print("Model konvertovan u ./rpi/model.onnx")