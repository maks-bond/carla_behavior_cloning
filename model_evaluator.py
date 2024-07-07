import torch

from model import DrivingModel

class ModelEvaluator:
    def __init__(self, model_path):
        self.model = DrivingModel()

        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def eval(self, features):
        features_tensor = torch.tensor(features, dtype=torch.float32)\

        output = self.model(features_tensor)
        return output.detach().numpy()