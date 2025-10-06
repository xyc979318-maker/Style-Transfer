import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt


def try_gpu():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
content_img = Image.open('初音.jpg').convert('RGB')
plt.imshow(content_img)
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 3, 2)
style_img = Image.open('梵高自画像.jpg').convert('RGB')
plt.imshow(style_img)
plt.title('Style Image')
plt.axis('off')

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img,img_shape):
    transforms=torchvision.transforms.Compose([torchvision.transforms.Resize(img_shape),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(rgb_mean,rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img=img[0].to(rgb_std.device)
    img=torch.clamp(img.permute(1,2,0)*rgb_std+rgb_mean,0,1)
    return torchvision.transforms.ToPILImage()(img.permute(2,0,1))

pretrained_net =torchvision.models.vgg19(pretrained=True)
style_layers,content_layers=[0,5,10,19,28],[25]
net=nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers+style_layers)+1)])

def extract_features(X,content_layers,style_layers):
    contents=[]
    styles=[]
    for i in range(len(net)):
        X=net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents,styles

def get_contents(image_shape,device):
    content_X=preprocess(content_img,image_shape).to(device)
    content_Y,_=extract_features(content_X,content_layers,style_layers)
    return content_X,content_Y
def get_styles(image_shape,device):
    style_X=preprocess(style_img,image_shape).to(device)
    _,style_Y=extract_features(style_X,content_layers,style_layers)
    return style_X,style_Y

def content_loss(Y_hat,Y):
    return torch.square(Y_hat-Y.detach()).mean()

def gram(X):
    num_chanels,n=X.shape[1], X.numel()//X.shape[1]
    X=X.reshape((num_chanels,n))
    return torch.matmul(X,X.T)/(num_chanels*n)

def style_loss(Y_hat,gram_Y):
    return torch.square(gram(Y_hat)-gram_Y.detach()).mean()

def tv_loss(Y_hat):
    return 0.5*(torch.abs(Y_hat[:,:,1:,:]-Y_hat[:,:,:-1,:]).mean()
                +torch.abs(Y_hat[:,:,:,1:]-Y_hat[:,:,:,:-1]).mean())

content_weight,style_weight,tv_weight=1,1e3,10

def compute_loss(X,contents_Y_hat,styles_Y_hat,content_Y,style_Y_gram):
    content_l=[content_loss(Y_hat,Y)*content_weight for Y_hat,Y in zip(contents_Y_hat,content_Y)]
    style_l=[style_loss(Y_hat,Y)*style_weight for Y_hat,Y in zip(styles_Y_hat,style_Y_gram)]
    tv_l=tv_loss(X)*tv_weight
    l=sum(10*style_l+content_l+5*[tv_l])  #书中的权重在此会导致出现较多的噪点
    return content_l,style_l,tv_l,l

class SynthesizedImage(nn.Module):
    def __init__(self,img_shape,**kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight=nn.Parameter(torch.rand(*img_shape))
    def forward(self):
        return self.weight

def get_inits(X,device,lr,styles_Y):
    gen_img=SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer=torch.optim.Adam(gen_img.parameters(),lr=lr)
    styles_Y_gram=[gram(Y) for Y in styles_Y]
    return gen_img,styles_Y_gram,trainer

def train(X,content_Y,style_Y,device,lr,num_epochs,lr_decay_epoch):
    X,style_Y_gram,trainer=get_inits(X,device,lr,style_Y)
    scheduler=torch.optim.lr_scheduler.StepLR(trainer,lr_decay_epoch,0.8)
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat,styles_Y_hat=extract_features(X(),content_layers,style_layers)
        contents_l,styels_l,tv_l,l=compute_loss(X(),contents_Y_hat,styles_Y_hat,content_Y,style_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}')
    return X()


def save_final_result(tensor, filename):
    img = postprocess(tensor)
    img.save(filename, quality=95)
    print(f"最终图像已保存: {filename}")

device,image_shape=try_gpu(),(1282,960)
net=net.to(device)
content_X,content_Y=get_contents(image_shape,device)
_,styles_Y=get_styles(image_shape,device)
output=train(content_X,content_Y,styles_Y,device,0.3,200,50)

output_img = postprocess(output.detach().cpu())
plt.subplot(1, 3, 3)
plt.imshow(output_img)
plt.title('Stylized Image')
plt.axis('off')
plt.tight_layout()
plt.show()

save_final_result(output, "风格迁移图像.jpg")