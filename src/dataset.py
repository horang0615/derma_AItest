from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os

class FolderBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.ds = ImageFolder(root=root_dir, transform=transform)
        # ImageFolder가 내부적으로 samples에 (path, label) 들고 있음
        self.samples = self.ds.samples

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        path, _ = self.samples[idx]
        image_id = os.path.basename(path)  # 파일명만
        return img, label, image_id
