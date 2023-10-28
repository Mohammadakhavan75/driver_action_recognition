from torchvision.transforms import transforms
import albumentations as A
from albumentations.augmentations.blur import transforms as a_t
from albumentations.core.composition import SomeOf 
class preprocessing:
    def __init__(self, image_size, num_transfroms=3):
        self.image_size = image_size
        self.num_transfroms = num_transfroms

    def loading_transforms(self):
        img_transforms_torch = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_augmentations = A.Compose([
            A.Resize(self.image_size, self.image_size, p=1),
            a_t.GaussianBlur(
                blur_limit=(3, 3), sigma_limit=0, always_apply=False, p=0.5),
            A.augmentations.transforms.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                brightness_by_max=True,
                always_apply=False,
                p=0.5),
            A.augmentations.transforms.RandomGamma(
                gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
            A.augmentations.transforms.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                                                    num_shadows_lower=1,
                                                    num_shadows_upper=2,
                                                    shadow_dimension=5,
                                                    always_apply=False,
                                                    p=0.5),
            A.augmentations.transforms.ColorJitter(brightness=0.1,
                                                contrast=0.1,
                                                saturation=0.2,
                                                hue=0.2,
                                                always_apply=False,
                                                p=0.5),
            A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0),
                                                mean=0,
                                                per_channel=True,
                                                always_apply=False,
                                                p=0.5),
            A.augmentations.transforms.HueSaturationValue(hue_shift_limit=20,
                                                        sat_shift_limit=30,
                                                        val_shift_limit=20,
                                                        always_apply=False,
                                                        p=0.5),
        ])

        img_augmentations = SomeOf(img_augmentations, n=self.num_transfroms)

        return img_transforms_torch, img_augmentations
