from torchvision import transforms
from PIL import Image

class CropInto4Transform:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image, mask = sample
        crops = []
        masks = []
        
        for i in range(2):
            for j in range(2):
                left = j * self.size
                upper = i * self.size
                right = left + self.size
                lower = upper + self.size
                
                cropped_image = image.crop((left, upper, right, lower))
                cropped_mask = mask.crop((left, upper, right, lower))
                
                cropped_image = transforms.ToTensor()(cropped_image)
                cropped_mask = transforms.ToTensor()(cropped_mask)

                crops.append(cropped_image)
                masks.append(cropped_mask)
        
        return crops, masks