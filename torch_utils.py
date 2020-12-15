import io 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image 

# 1 load our saved model

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

input_size=784
hidden_size=500
num_classes=10
model = NeuralNet(input_size, hidden_size, num_classes)

path = "mnist_nn.pth"
model.load_state_dict(torch.load(path))
model.eval()

# 2 Image -> Tensor
def transform_imgae(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0) 
    # 이미지가 하나밖에 없기 때문에 batch의 축을 unsqueeze해줌 

# 3 predict 
def  get_prediction(image_tensor): 
    images = image_tensor.reshape(-1,28*28)
    outputs = model(images)
        # max returns (value, index)
    _, pred = torch.max(outputs.data, 1)
        # torch.max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
            # input (Tensor) – the input tensor.
            # dim (int) – the dimension to reduce.
            # keepdim (bool) – whether the output tensor has dim retained or not. Default: False.
    return pred


""" SAVING ON GPU/CPU 

# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# 3) Save on CPU, Load on GPU

torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""