import os
from PIL import Image
from torch.utils.data import Dataset


def extract_images(root):
    """ Extract all images under root """
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    root = os.path.expanduser(root)
    img_paths = []
    for curdir, subdirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_ext:
                img_paths.append(os.path.join(curdir, file))
    img_paths = sorted(img_paths)
    return img_paths


class ImageDir(Dataset):
    def __init__(self, root, transform=None):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise ValueError(f'{root} is not a valid directory')

        self.transform = transform
        self.img_paths = extract_images(root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X
