import torch
from torchvision import transforms

def eval(model,
         img,
         gpu_f=True,
         transform=transforms.Compose([transforms.Resize((300, 300)),transforms.ToTensor()])
         ):
    """
    :param model: model to be tested
    :param img: img for testing
    :return: return predictions drawn on image
    """
    model.eval()
    data = transform(img).unsqueeze(dim=0)

    with torch.no_grad():
        if gpu_f:
            data = data.to(torch.device('cuda:0'))
        out = model(data)
