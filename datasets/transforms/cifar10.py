from torchvision import transforms
import torchvision.transforms as T

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

def cifar10_train():
    transform = transforms.Compose([
        T.RandomResizedCrop((32, 32)),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform

def cifar10_valid():
    transform = transforms.Compose([
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform