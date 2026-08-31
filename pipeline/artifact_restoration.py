import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from core.util import set_device
from models.network import Network

class RestorationModel:
    def __init__(self, config):
        self.config = config
        model_args = config["config_restoration"]["model"]["which_networks"][0]["args"]
        model_path = config["restoration_model_path"]
        self.model = Network(**model_args)

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = set_device(self.model)
        self.model.set_new_noise_schedule(phase='test')
        self.model.eval()
        print("Load Model Palette")
        
        imgsize = model_args['unet']['image_size']
        self.transform_img = transforms.Compose( [transforms.Resize((imgsize,imgsize), transforms.InterpolationMode.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]
                            )

        self.transform_mask = transforms.Compose( [
                                              transforms.Resize((imgsize,imgsize), transforms.InterpolationMode.BICUBIC),
                                              transforms.ToTensor()
                                        ]
                                    )
        
    def preprocess_image(self, image_path=None, image=None):
        if image is not None:        
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                img = image.copy()
            else:
                raise TypeError(f"not support image type: {type(image)}")
        elif image_path is not None:        
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"not exist file: {image_path}")
            img = Image.open(image_path).convert('RGB')
            if img is None:
                raise ValueError(f"can not read file: {image_path}") 
        else:
            raise ValueError("please provide the image data in one of the following formats: a file path (image_path), a Image.Image or a NumPy array (image).")
        if img.size == 0:
            raise ValueError("the image is empty")
        return img
        
    def preprocess(self, image, transform):
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        return image
    
    def postprocess(self, image):
        image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        image = image.permute(0, 2, 3, 1)
        image = image.contiguous().cpu().numpy()
        return image
        
    def inpaint(self, image_path=None, image=None, mask_path=None, mask=None, use_jump=True, resample=3, jump_length=10, progress=True, cond_fn=None, fn_kywards=None, gradient_fn=None):
        img = self.preprocess_image(image_path, image)
        mask = self.preprocess_image(mask_path, mask)
        
        img_tensor = self.preprocess(img, self.transform_img)
        mask_tensor = self.preprocess(mask, self.transform_mask)
        img_tensor = set_device(img_tensor)
        mask_tensor = set_device(mask_tensor)

        cond_image = img_tensor*(1. - mask_tensor) + mask_tensor*torch.randn_like(img_tensor)
        cond_image = cond_image

        with torch.no_grad():
            if use_jump:
                output = self.model.restoration_with_jump_sampling(cond_image, y_t=cond_image, y_0=img_tensor, mask=mask_tensor,
                                                                   resample=resample, jump_length=jump_length, progress=progress,
                                                                   cond_fn=cond_fn, fn_kywards=fn_kywards, gradient_fn=gradient_fn)
            else:
                output, visuals = self.model.restoration(cond_image, y_t=cond_image, y_0=img_tensor, mask=mask_tensor, 
                                                         progress=progress,
                                                         cond_fn=cond_fn, fn_kywards=fn_kywards, gradient_fn=gradient_fn)
            
        output = self.postprocess(output)
            
        result = {}
        result['image'] = img
        result['mask'] = mask
        result['restore'] = output[0]
        return result
    
    def inpaint_batch(self, img, mask, use_jump=True, resample=3, jump_length=10, progress=True, cond_fn=None, fn_kywards=None, gradient_fn=None):
        img = set_device(img)
        mask = set_device(mask)

        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        cond_image = cond_image

        with torch.no_grad():
            if use_jump:
                output = self.model.restoration_with_jump_sampling(cond_image, y_t=cond_image, y_0=img, mask=mask,
                                                                   resample=resample, jump_length=jump_length, progress=progress,
                                                                   cond_fn=cond_fn, fn_kywards=fn_kywards, gradient_fn=gradient_fn)
            else:
                output, visuals = self.model.restoration(cond_image, y_t=cond_image, y_0=img, mask=mask, 
                                                         progress=progress,
                                                         cond_fn=cond_fn, fn_kywards=fn_kywards, gradient_fn=gradient_fn)

        return output