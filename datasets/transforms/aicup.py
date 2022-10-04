from torchvision import transforms
import torchvision.transforms as T

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

def base():
    transform = transforms.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
    ])
    return transform
