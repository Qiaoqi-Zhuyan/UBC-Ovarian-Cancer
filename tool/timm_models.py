import timm
import torch
from torch import nn

for model in timm.list_models():
   print(model)


model_ = timm.create_model(
   model_name='mobilevit_s',
   pretrained=True,
)
print(model_.default_cfg)


class Model(nn.Module):
   def __init__(self, model):
      super(Model, self).__init__()
      self.model = model
      self.classifier = nn.Linear(1000, 5)

   def forward(self, x):
      return self.classifier(self.model(x))

model = Model(model_)

x = torch.randn(1, 3, 224, 224)

print(model(x).shape)

