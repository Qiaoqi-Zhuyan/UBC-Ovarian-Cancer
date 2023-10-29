import torch
import torch.nn as nn
import timm



class UBCModel_ResNet50(nn.Module):
    def __init__(self, img_size):
        super(UBCModel_ResNet50, self).__init__()
        self.num_class = 5
        self.model = timm.create_model(
            model_name="resnet50",
            pretrained=None,
            in_chans = 3,
        )

        self.out_feature = self.model.fc.out_features
        self.fc1 = nn.Linear(self.out_feature, self.out_feature // 2)
        self.fc2 = nn.Linear(self.out_feature // 2, self.num_class)
        self.softmax = nn.Softmax( dim=1)


    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = UBCModel_ResNet50(1024)


m = torch.randn((1, 3,1024, 1024))

pred = model(m)



#print(model)
