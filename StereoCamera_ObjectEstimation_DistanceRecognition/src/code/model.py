import torch

def load_model(weights_path='/Users/apple/Desktop/Python/yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    model.eval()  # Set model to evaluation mode
    return model
