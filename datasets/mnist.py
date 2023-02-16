import torchvision.transforms as T
import torchvision.datasets as dset


MNIST = dset.MNIST


def get_default_transforms(img_size: int):
    transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    return transforms
