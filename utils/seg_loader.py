from torchvision.datasets.folder import default_loader
from torchvision.datasets import DatasetFolder

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class SegmentationFolder(DatasetFolder):

    def __init__(
            self,
            root: str,
            root_mask: str,
            transform,
            target_transform,
            mask_transform,
            loader = default_loader,
            is_valid_file = None,
    ):
        super(SegmentationFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        classes, class_to_idx = self.find_classes(self.root)
        self.extensions = extensions
        self.root_mask = root_mask
        self.mask_transform = mask_transform
        mask = self.make_dataset(self.root_mask, class_to_idx, extensions, is_valid_file)
        self.mask = mask
        self.imgs = self.samples

    def __getitem__(self, index: int) :
        
        path, target = self.samples[index]
        mask_path, _ = self.mask[index]
        sample = self.loader(path)
        mask = self.loader(mask_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask.convert('L'))

        return sample, target, mask
