from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Compose, ImageOnlyTransform
from albumentations.augmentations import functional as F
import cv2

class ResizeToFixedHeight(ImageOnlyTransform):
    """Resize image to have fixed height while keeping aspect ratio"""
    def __init__(self, height, min_width=None, max_width=None):
        super().__init__(p=1.0)
        self.height = height
        self.min_width = min_width
        self.max_width = max_width

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        height, width, _ = img.shape
        target_width = width * self.height // height
        if self.min_width:
            target_width = max(target_width, self.min_width)
        if self.max_width:
            target_width = min(target_width, self.max_width)

        return F.resize(img, height=self.height, width=target_width, interpolation=interpolation)

def pre_transforms(height, min_width=None, max_width=None):
    """Transforms which should always be applied before others"""
    return ResizeToFixedHeight(height, min_width, max_width)

def post_transforms():
    """Transforms whild should always be applied after others"""
    return Compose([
        Normalize(),
        ToTensorV2()
    ])
