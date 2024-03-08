# write a script to load the trained model and make predictions on the test set
#
import torch
from tqdm import tqdm
from torchvision import transforms
from torch import nn
from PIL import Image
from torchvision import models
from torchvision import transforms


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

class ResNetClassifier:
    def __init__(self, model_path, device=0):
       
        self.model = ResNet18(pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.class_names = [ 'negative', 'flag']
        
    def predict_one_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
        _, predicted = torch.max(output.data, 1)
        # confidence
        confidence = torch.max(output).item()
        # if predicted.item() == 0 and confidence < 0.6:
        #     return "flag"
        print("confidence:",confidence )
        return self.class_names[predicted.item()]
    
    def predict_one_image_path(self, image_path):
        image = Image.open(image_path)
        return self.predict_one_image(image)

    def predict_a_batch(self, image_list):
        image_list = [self.transform(image) for image in image_list]
        image_list = torch.stack(image_list)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            output = self.model(image_list)
            #print(output)
        _, predicted = torch.max(output.data, 1)
        return [self.class_names[predicted.item()] for predicted in predicted]


if __name__ == "__main__":
    classifier = ResNetClassifier(model_path=r"E:\HD_VNese_map\dataset\flag_classify\best_model.pth")
    print(classifier.predict_one_image_path(r"1.png"))