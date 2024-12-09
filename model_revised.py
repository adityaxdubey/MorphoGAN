import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as tf
from torchvision.utils import save_image
from torchvision.models import vgg19
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def AdaIN(content_features,style_features,eps=1e-5):
    if content_features.shape[2:]!=style_features.shape[2:]:
        style_features=F.interpolate(style_features,size=content_features.shape[2:],mode='nearest')
    
    #normalization,matches the mean and variance of content features to the style features
    content_mean,content_std=content_features.mean([2,3],keepdim=True),content_features.std([2,3],keepdim=True)
    style_mean,style_std=style_features.mean([2,3],keepdim=True),style_features.std([2,3],keepdim=True)
    normalized_content=(content_features-content_mean)/(content_std+eps)
    return normalized_content*style_std+style_mean


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        vgg=vgg19(pretrained=True).features
        self.layers=nn.Sequential(*[vgg[i] for i in range(21)])  # Up to relu4_1
    
    def forward(self,x):
        features=[]
        for layer in self.layers:
            x=layer(x)
            features.append(x)
        return features  # Extract features from different layers

#skip connections help model learn identity mappings and prevent vanishing gradients
class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(in_channels),
        )
    def forward(self,x):
        return x+self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.model=nn.Sequential(
            ResidualBlock(512),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1),
        )
    
    def forward(self,x):
        return self.model(x)

def content_loss(content_features,transformed_features):
    if content_features.shape[2:]!=transformed_features.shape[2:]:
        transformed_features=F.interpolate(transformed_features,size=content_features.shape[2:],mode='bilinear',align_corners=False)
    return F.mse_loss(content_features,transformed_features)

#used gram matrix coz it helps to retain the relationships between different features in the image and is compared with generated image
def gram_matrix(features):
    (b,c,h,w)=features.size()
    features=features.view(b,c,h*w)
    gram=torch.bmm(features,features.transpose(1,2))
    return gram/(c*h*w)

def style_loss(style_features,transformed_features):
    loss=0
    for sf,tf in zip(style_features,transformed_features):
        loss+=F.mse_loss(gram_matrix(sf),gram_matrix(tf))
    return loss

def total_loss(content_features,transformed_features,style_features,transformed_style_features,content_weight=1.0,style_weight=1e4):
    c_loss=content_loss(content_features[-1],transformed_features[-1])
    s_loss=style_loss(style_features,transformed_style_features)
    return content_weight*c_loss+style_weight*s_loss

#below are image translations fn
def load_image(image_path,image_size=256):
    transform=tf.Compose([
        tf.Resize((image_size,image_size)),
        tf.ColorJitter(brightness=0.2,contrast=0.2),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    image=Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def denormalize_image(tensor):
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(tensor.device)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(tensor.device)
    return tensor*std+mean

def save_output_image(tensor,output_path):
    tensor=denormalize_image(tensor)
    save_image(torch.clamp(tensor,0,1),output_path)

def train_style_transfer(content_image_path,style_image_path,output_image_path,image_size=256,epochs=500,content_weight=1.0,style_weight=1e4,lr=1e-4):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder=Encoder().to(device).eval()
    decoder=Decoder().to(device)
    optimizer=optim.Adam(decoder.parameters(),lr=lr)

    content_image=load_image(content_image_path,image_size).to(device)
    style_image=load_image(style_image_path,image_size).to(device)

    for epoch in tqdm(range(epochs),desc="Training"):
        optimizer.zero_grad()

        content_features=encoder(content_image)
        style_features=encoder(style_image)

        #it generates transformed features
        content_style_features=AdaIN(content_features[-1],style_features[-1])
        transformed_image=decoder(content_style_features)

        transformed_features=encoder(transformed_image)
        #time to compute loss :>
        c_loss=content_loss(content_features[-1],transformed_features[-1])
        s_loss=style_loss(style_features,transformed_features)
        t_loss=total_loss(content_features,transformed_features,style_features,transformed_features,content_weight,style_weight)

        #going backwards(propagation)
        t_loss.backward()
        optimizer.step()

        if (epoch+1)%50==0:
            print(f"Epoch {epoch+1}, Total Loss: {t_loss.item()}")

    save_output_image(transformed_image,output_image_path)
    print(f"Saved styled image to {output_image_path}")

if __name__=="__main__":
    content_image_path="input/input1.jpg"
    style_image_path="style/style1.jpg"
    output_image_path="output/output1_revised1.jpg"

    train_style_transfer(content_image_path,style_image_path,output_image_path,epochs=500)

