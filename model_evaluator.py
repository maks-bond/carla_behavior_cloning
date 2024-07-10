import torch

from model import DrivingModel

class ModelEvaluator:
    def __init__(self, model_path):
        self.model = DrivingModel()

        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def eval(self, features):
        # Convert features list to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Reshape the tensor to have a batch dimension
        features_tensor = features_tensor.unsqueeze(0)

        output = self.model(features_tensor)
        output = output.squeeze(0)
        return output.detach().numpy()