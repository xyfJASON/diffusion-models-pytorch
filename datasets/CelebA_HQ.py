import os
from PIL import Image
from torch.utils.data import Dataset


class CelebA_HQ(Dataset):
    """
    The downloaded 30,000 images should be stored directly under `root`.

    The file names should be the same as their counterparts in the original CelebA dataset.

    The train/valid/test sets are split according to the original CelebA dataset,
    resulting in 24,183 training images, 2,993 validation images, and 2,824 test images.

    """
    def __init__(self, root, split='train', transform=None):
        assert os.path.isdir(root)
        assert split in ['train', 'valid', 'test', 'all']

        self.transform = transform

        img_ext = ['.png', '.jpg', '.jpeg']
        self.img_paths = []
        for curdir, subdirs, files in os.walk(root):
            for file in files:
                if os.path.splitext(file)[1].lower() in img_ext:
                    self.img_paths.append(os.path.join(curdir, file))
        self.img_paths = sorted(self.img_paths)

        celeba_splits = [1, 162771, 182638, 202600]

        def filter_func(p):
            if split == 'all':
                return True
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.img_paths = list(filter(filter_func, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        return X


if __name__ == '__main__':
    dataset = CelebA_HQ(root='/Users/jason/data/celeba-hq/CelebA-HQ-img', split='train')
    print(len(dataset))
    dataset = CelebA_HQ(root='/Users/jason/data/celeba-hq/CelebA-HQ-img', split='valid')
    print(len(dataset))
    dataset = CelebA_HQ(root='/Users/jason/data/celeba-hq/CelebA-HQ-img', split='test')
    print(len(dataset))
    dataset = CelebA_HQ(root='/Users/jason/data/celeba-hq/CelebA-HQ-img', split='all')
    print(len(dataset))
