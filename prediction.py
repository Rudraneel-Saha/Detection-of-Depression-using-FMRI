import torch
from models.train import DepressionCNN
from models.train import preprocess_nifti

def load_model():
    model = DepressionCNN()
    model.load_state_dict(torch.load("depression_cnn.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

def predict_nifti(path):
    tensor = preprocess_nifti(path).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1).item()
    return pred  # 0 = control, 1 = depressed
