from torchvision import models
from torch import nn

class get_models:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers

    def get_model(self):
        if self.name == "resnet18":
            model = models.resnet18(pretrained=True)
            num_ftrs = model.classifier[1].in_features
        
        if self.name == "resnet34":
            model = models.resnet34(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "resnet50":
            model = models.resnet50(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "resnet101":
            model = models.resnet101(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "resnet152":
            model = models.resnet152(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "vit_b_16":
            model = models.vit_b_16(pretrained=True)
            num_ftrs = model.classifier[1].in_features
        
        if self.name == "vit_b_32":
            model = models.vit_b_32(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "vit_l_16":
            model = models.vit_l_16(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "vit_l_32":
            model = models.vit_l_32(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "vit_h_14":
            model = models.vit_h_14(pretrained=True)
            num_ftrs = model.classifier[1].in_features
        
        if self.name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_v2_s":
            model = models.efficientnet_v2_s(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_v2_m":
            model = models.efficientnet_v2_m(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_v2_l":
            model = models.efficientnet_v2_l(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b0":
            model = models.efficientnet_b0(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b1":
            model = models.efficientnet_b1(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b2":
            model = models.efficientnet_b2(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b3":
            model = models.efficientnet_b3(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b4":
            model = models.efficientnet_b4(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b5":
            model = models.efficientnet_b5(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b6":
            model = models.efficientnet_b6(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if self.name == "efficient_b7":
            model = models.efficientnet_b7(pretrained=True)
            num_ftrs = model.classifier[1].in_features

        if len(self.layers) > 1:
            model.classifier[1] = nn.Linear(num_ftrs, self.layers[0])
            for i in range(len(self.layers)-1):
                model.classifier.append(nn.Dropout(p=0.2))
                model.classifier.append(nn.Linear(self.layers[i], self.layers[i+1]))
        else:
            model.classifier[1] = nn.Linear(num_ftrs, self.layers[0])
            
        return model