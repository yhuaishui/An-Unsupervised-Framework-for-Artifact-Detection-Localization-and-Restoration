import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.multiprocessing as mp

import numpy as np
from PIL import Image
import cv2
import glob
import skimage

from models import fastflow
from pipeline.parser import load_config

def load_fastflow_model(args):
    model = fastflow.build_model(args['fastflow'], args['fastflow']['backbone_pretrained'])
    model.cuda()
    checkpoint = torch.load(args['fastflow']['resume_and_normalize_config']['resume_path'])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_transform(input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return image_transform

def preprocess_image(image_path=None, image=None, image_size=256):
    if image is not None:        
        if isinstance(image, np.ndarray):
            img = image.copy()
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise TypeError(f"not support image type: {type(image)}")
    elif image_path is not None:        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"not exist file: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        if img is None:
            raise ValueError(f"can not read file: {image_path}") 
    else:
        raise ValueError("please provide the image data in one of the following formats: a file path (image_path), a Image.Image or a NumPy array (image).")
    if img.size == 0:
        raise ValueError("the image is empty")
        
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = Image.fromarray(img)
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img = np.array(img)
    return img

def normalize_map(
    targets, threshold, min_val, max_val
):
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    if threshold != 0 :
        normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    else:
        normalized = ((targets - min_val) / (max_val - min_val)) 
    if isinstance(targets, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized

class UnsegNet(nn.Module):
    def __init__(self, inp_dim=3, mod_dim1=64, mod_dim2=32):
        super(UnsegNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )
    def forward(self, x):
        return self.seq(x)
    
def UnsegRun(config, image, image_tensor, seg_lab, device):
    """Perform unsupervised segmentation on an input image using a neural network"""
    # train init
    model = UnsegNet(inp_dim=config['inp_dim'], mod_dim1=config['mod_dim1'], mod_dim2=config['mod_dim2']).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    image_flatten = image.copy().reshape((-1, 3))

    # train loop
    model.train()
    for batch_idx in range(config['iteration']):
        # forward
        optimizer.zero_grad()
        output = model(image_tensor)[0]
        output = output.permute(1, 2, 0).view(-1, config['mod_dim2'])
        target = torch.argmax(output, 1)
        
        # refine
        im_target = target.data.cpu().numpy()
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]
        target = torch.from_numpy(im_target)
        target = target.to(device)
            
        # backward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Visualize when categories < min threshold or at final iteration  
        un_label, lab_inverse = np.unique(im_target, return_inverse=True)
        if len(un_label) < config['min_category']:
            color_avg = [np.mean(image_flatten[im_target == label], axis=0, dtype=np.int32) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                image_flatten[lab_inverse == lab_id] = color
            show = image_flatten.reshape(image.shape)
            break
        if batch_idx == config['iteration']-1:
            color_avg = [np.mean(image_flatten[im_target == label], axis=0, dtype=np.int32) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                image_flatten[lab_inverse == lab_id] = color
            show = image_flatten.reshape(image.shape)

    return show, color_avg

def calculate_mask_stats(mask, prob_map):
    """Compute average probability and pixel area for each mask"""
    assert mask.shape == prob_map.shape, "Mask and probability map have mismatched dimensions"
    masked_probs = prob_map[mask > 0] 
    avg_prob = np.mean(masked_probs) if masked_probs.size > 0 else 0.0
    area = np.sum(mask > 0)
    return avg_prob, area

def Filter_UnSegMasks(prob_map, all_segmentations,
                      prob_diff=0.03, max_area_percentage=0.5, min_area_percentage=0.01, prob_high_confidence=0.6):
    """Stage I: For each segmentation, compute the max probability and its associated area"""
    max_records = []
    for seg in all_segmentations:
        candidates = []
        for mask in seg:
            avg_prob, area = calculate_mask_stats(mask, prob_map)
            # Remove regions with areas that do not meet the requirements
            h, w = prob_map.shape[:2]
            max_pixels = h * w * max_area_percentage
            min_pixels = h * w * min_area_percentage
            if area > max_pixels or area < min_pixels:
                continue
            candidates.append((avg_prob, area))

        # Handle cases with no candidate regions
        if not candidates:
            max_records.append((0.0, 0))
            continue

        # Determine the max probability and the corresponding threshold range
        max_prob = max(p for p, _ in candidates)
        prob_lower_bound = max_prob - prob_diff
        # Filter probability candidates and choose the largest one by area
        high_prob_candidates = [
            (p, a) for p, a in candidates if p >= prob_lower_bound
        ]
        best_avg_prob, best_area = max(high_prob_candidates, key=lambda x: x[1])
        max_records.append((best_avg_prob, best_area))
    
        
    """Stage II: For max_records in each segmentation,sort and choose the best segmentation result"""
    sorted_indices = sorted(
        range(len(max_records)),
        key=lambda i: (-max_records[i][0], -max_records[i][1]),
    )
    
    to_out = []
    top_prob = max_records[sorted_indices[0]][0]
    for idx in sorted_indices:
        current_prob, current_area = max_records[idx]
        if abs(current_prob - top_prob) <= prob_diff:
            to_out.append(idx)
        else:
            break
    
    if len(to_out) > 1:
        max_area = -1
        final_idx = to_out[0]
        for idx in to_out:
            if max_records[idx][1] > max_area:
                max_area = max_records[idx][1]
                final_idx = idx
        to_out = [final_idx]
    else:
        to_out = [sorted_indices[0]]
        
        
    """Stage III: Filter the candidates in the best segmentation result"""
    max_prob = max_records[to_out[0]][0]
    max_area = max_records[to_out[0]][1]
    mask_out=np.zeros_like(prob_map)
    max_prob_mask_out=np.zeros_like(prob_map)
    h, w = prob_map.shape[:2]
    max_pixels = h * w * max_area_percentage
    min_pixels = h * w * min_area_percentage
    for mask in all_segmentations[to_out[0]]:
        avg_prob, area = calculate_mask_stats(mask, prob_map)
        if area > max_pixels or area < min_pixels:
            continue
        if np.isclose(avg_prob, max_prob, atol=prob_diff) or avg_prob > prob_high_confidence:
            mask_out += mask
        if avg_prob == max_prob:
            max_prob_mask_out = mask
            
    mask_out = np.where(mask_out > 0, 1, 0).astype(np.float32)
    max_prob_mask_out = np.where(max_prob_mask_out > 0, 1, 0).astype(np.float32)
    return mask_out, max_prob_mask_out, to_out[0]

def auto_detect_background(image, lower=[0, 0, 200], upper=[180, 30, 255]):
    """
    Automatically detect background regions in an image using HSV color range thresholds
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array(lower, dtype="uint8")  
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def background_score_func(true, pred):
    """Compute a simplified IoU variant"""
    true = (true > 127).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)
    
    intersection = (pred * true).sum()    
    score = intersection / (pred.sum()+0.0001)
    
    return round(score,3)

def filter_components_by_size(mask, min_size_ratio=0.02, max_area_ratio=0.1):
    """Filter connected components based on the area"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S
    )
    

    max_area, max_label = 0, 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area, max_label = area, label   
            
    min_size = mask.size * min_size_ratio
    max_threshold = max_area * max_area_ratio
    processed_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > min_size and area > max_threshold:
            processed_mask[labels == label] = 1

    if np.all(processed_mask == 0):
        processed_mask = np.where(labels == max_label, 1, 0)
        
    return processed_mask.astype(np.uint8)

def filter_components_by_map_overlap(mask, map_mask, overlap_threshold=0.1):
    """Filter connected components based on the overlap area, retaining components that overlap sufficiently with the reference region."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S
    )
    
    processed_labels = np.zeros_like(mask).astype(np.float32)
    for label in range(1, num_labels):#
        component_mask  = np.where(labels == label, 1, 0).astype(np.float32)
        area = stats[label, cv2.CC_STAT_AREA]
        overlap_area  = np.sum(component_mask * map_mask)
        overlap_ratio = overlap_area / area
        if overlap_area > overlap_threshold:
            processed_labels += component_mask

    processed_labels = np.where(processed_labels > 0, 1, 0).astype(np.float32)  
    return processed_labels

def extract_largest_components(mask, num_components=2):
    """
    Extract the N largest connected components.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S
    )    
    areas = [(stats[label, cv2.CC_STAT_AREA], label) for label in range(1, num_labels)]
    areas.sort(reverse=True)
    top_labels = [label for _, label in areas[:num_components]]
    processed_labels = np.isin(labels, top_labels)
    processed_labels = np.where(processed_labels>0, 1, 0).astype(np.float32)
    return processed_labels

def count_components(mask, min_area=30):
    """Count the number of connected components that meet a minimum area requirement."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S
    )
    
    valid_component_count  = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            valid_component_count +=1
    return valid_component_count 

def fill_and_smooth_mask(mask, kernel_size=5, open_iterations=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = cv2.drawContours(cleaned.copy(), contours, -1, 255, thickness=cv2.FILLED)
    
    return filled

def combine_mask(config, image, anomaly_map, un_seg_mask, un_seg_max_prob_mask):
    """Combine unsupervised segmentation mask and anomaly heatmap with background detection to generate a high‑quality final defect localization mask."""
    valid_component_count = count_components(un_seg_mask)
    mask_ratio = np.count_nonzero(un_seg_mask) / un_seg_mask.size
    # Unseg initial is too large or too cluttered; use the heatmap instead
    if mask_ratio > config['localization']['max_area_percentage'] or valid_component_count > config['localization']['filter_component_count']:
        un_seg_mask = anomaly_map > 0.5
        
    kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    background = auto_detect_background(image, lower=config['background']['lower_hsv'], upper=config['background']['upper_hsv'])
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel_5)
    background = cv2.dilate(background, kernel_5, iterations=2)
    
    # combine unseg
    max_components = cv2.erode(un_seg_max_prob_mask, kernel_5, iterations=2) 
    max_components = extract_largest_components(max_components)
    background_cleaned = background * (~(max_components>0))
    background_cleaned = np.where(background_cleaned > 0, 1, 0).astype(np.float32)
    if np.sum(background_cleaned) / background_cleaned.size < config['background']['background_purity_threshold']:
        un_seg_mask = un_seg_mask - background_cleaned
    else:
        un_seg_mask = extract_largest_components(un_seg_mask)
    un_seg_mask = np.where(un_seg_mask > 0, 1, 0).astype(np.uint8)
    un_seg_mask = cv2.dilate(un_seg_mask, kernel_3, iterations=2)
    un_seg_mask = filter_components_by_size(un_seg_mask)
    un_seg_mask = cv2.dilate(un_seg_mask, kernel_7, iterations=1)
    un_seg_mask = fill_and_smooth_mask(un_seg_mask)
    un_seg_mask = np.where(un_seg_mask > 0, 1, 0).astype(np.float32)
    
    # combine anomaly_map mask
    anomaly_map_mask = anomaly_map>config['localization']['combine_map_threshold']
    map_mask = extract_largest_components(anomaly_map_mask)
    map_mask = np.where(map_mask > 0, 1, 0).astype(np.float32)
    map_max_mask = extract_largest_components(anomaly_map_mask)
    background_cleaned = background * (~(map_max_mask>0))
    background_cleaned = np.where(background_cleaned > 0, 1, 0).astype(np.float32)
    if np.sum(background_cleaned) / background_cleaned.size < config['background']['background_purity_threshold']:
        map_mask = map_mask - background_cleaned
    else:
        map_mask = extract_largest_components(map_mask)
    map_mask = np.where(map_mask > 0, 1, 0).astype(np.uint8)
    map_mask = filter_components_by_size(map_mask)
    map_mask = cv2.dilate(map_mask, kernel_3, iterations=2)
    map_mask = fill_and_smooth_mask(map_mask)
    map_mask = np.where(map_mask > 0, 1, 0).astype(np.float32)
    
    # combine
    final_mask = un_seg_mask + map_mask
    final_mask = np.where(final_mask > 0, 1, 0).astype(np.uint8)
    final_mask = filter_components_by_map_overlap(final_mask, anomaly_map_mask)
    return final_mask


class HisAnomalyModel:
    def __init__(self, 
                 config = 'config/config_detection_localization.json',
                ):
        self.config = load_config(config)
        self.config['fastflow']['resume_and_normalize_config'] = load_config(self.config['fastflow']['resume_and_normalize_config_path'])
        self.model = load_fastflow_model(self.config)
        self.image_transform = get_transform(self.config['fastflow']['input_size'], self.config['fastflow']['transforms_mean'], self.config['fastflow']['transforms_std'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        
    
    def detection_and_localization(self, image_path=None, image=None, detail=False):
        # predict anomaly_map with fastflow
        image = preprocess_image(image_path, image, image_size=self.config['fastflow']['output_size'])
        image_tensor = self.image_transform(image)
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(dim=0).to(self.device)
            outputs = self.model(image_tensor)
        anomaly_map = outputs["anomaly_map"].cpu().detach().numpy()[0,0]
        anomaly_map = normalize_map(anomaly_map, self.config['fastflow']['resume_and_normalize_config']['pixel_threshold'], self.config['fastflow']['resume_and_normalize_config']['pixel_min'], self.config['fastflow']['resume_and_normalize_config']['pixel_max'])
        
        # detect with abnormal_pixel_percentage
        pred_mask = anomaly_map > 0.5
        area = np.sum(pred_mask > 0)
        prob = area / pred_mask.shape[0]**2
        pred_label = prob > self.config['detection']['abnormal_pixel_percentage']
        
        # detect with background
        if pred_label and self.config['background']['remove_background']:
            background = auto_detect_background(image)
            background = skimage.morphology.remove_small_objects(
                background > 0,
                min_size = 400,
                connectivity=1
            ).astype(np.uint8) * 255
            background_score = background_score_func(background, anomaly_map)
            # high background score: background_label_inv set False, no localization is performed
            background_label_inv = (background_score < self.config['background']['bg_ratio_threshold'])
            background_percentage = np.sum(background>0) / background.size
        else:
            # background is not excluded, locate all artifacts
            background_label_inv = True
            background_percentage = 0
            background_score = 0
            background = np.zeros_like(image)
        
        if pred_label and background_label_inv:
            # unsupervised image segmentation
            all_segmentations = []
            all_result = []
            all_color_avg = []
            # init image tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            # segmentation ML
            seg_map = skimage.segmentation.felzenszwalb(image, scale=self.config['localization']['felzenszwalb']['scale'], sigma=self.config['localization']['felzenszwalb']['sigma'], min_size=self.config['localization']['felzenszwalb']['min_size'])
            seg_map = seg_map.flatten()
            seg_lab = [np.where(seg_map == u_label)[0]
                       for u_label in np.unique(seg_map)]
            for i in range(self.config['localization']['number_seg_runs']):
                result, color_avg = UnsegRun(self.config['localization']['model'], image, image_tensor, seg_lab, self.device)
                segmentations = []
                for color_id in color_avg:
                    tmp = np.zeros_like(result)
                    mask = (result == color_id)
                    tmp[mask] = 1
                    tmp=tmp[...,0]
                    segmentations.append(tmp)
                all_segmentations.append(segmentations)
                all_result.append(result)
                all_color_avg.append(color_avg)

            un_seg_mask, un_seg_max_prob_mask, to_out_index = Filter_UnSegMasks(anomaly_map, all_segmentations, prob_diff=self.config['localization']['prob_diff'], max_area_percentage=self.config['localization']['max_area_percentage'], min_area_percentage=self.config['localization']['min_area_percentage'], prob_high_confidence=self.config['localization']['prob_high_confidence'])
                  
            # combine map mask
            mask = combine_mask(self.config, image, anomaly_map, un_seg_mask, un_seg_max_prob_mask)
        else:
            mask = np.zeros_like(anomaly_map)

        result={}
        result['image'] = image
        result['anomaly_map'] = anomaly_map
        result['pred_label'] = pred_label
        result['background'] = background
        result['background_label'] = not background_label_inv
        result['mask'] = mask
        if detail:
            result['background_percentage'] = background_percentage
            result['background_score'] = background_score
            result['refine_pred_label'] = np.sum(mask > 0) != 0
            result['mask_percentage'] = np.sum(mask > 0) / mask.size
            if pred_label and background_label_inv:
                result['show_unseg'] = all_result[to_out_index]
                result['show_color_avg'] = all_color_avg[to_out_index]
        return result
    
    def detection(self, image_path=None, image=None):
        image = preprocess_image(image_path, image, image_size=self.config['fastflow']['output_size'])
        image_tensor = self.image_transform(image)
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(dim=0).to(self.device)
            outputs = self.model(image_tensor)
        anomaly_map = outputs["anomaly_map"].cpu().detach().numpy()[0,0]
        anomaly_map = normalize_map(anomaly_map, self.config['fastflow']['resume_and_normalize_config']['pixel_threshold'], self.config['fastflow']['resume_and_normalize_config']['pixel_min'], self.config['fastflow']['resume_and_normalize_config']['pixel_max'])
        
        pred_mask = anomaly_map > 0.5
        area = np.sum(pred_mask > 0)
        prob = area / pred_mask.shape[0]**2
        pred_label = prob > self.config['detection']['abnormal_pixel_percentage']
        
        result={}
        result['image'] = image
        result['anomaly_map'] = anomaly_map
        result['pred_label'] = pred_label
        return result
    
    def localization(self, image_path=None, image=None, anomaly_map=None):
        image = preprocess_image(image_path, image, image_size=self.config['fastflow']['output_size'])

        if anomaly_map is None:
            image_tensor = self.image_transform(image)
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(dim=0).to(self.device)
                outputs = self.model(image_tensor)
            anomaly_map = outputs["anomaly_map"].cpu().detach().numpy()[0,0]
            anomaly_map = normalize_map(anomaly_map, self.config['fastflow']['resume_and_normalize_config']['pixel_threshold'], self.config['fastflow']['resume_and_normalize_config']['pixel_min'], self.config['fastflow']['resume_and_normalize_config']['pixel_max'])
        
        pred_mask = anomaly_map > 0.5
        area = np.sum(pred_mask > 0)
        prob = area / pred_mask.shape[0]**2
        pred_label = prob > self.config['detection']['abnormal_pixel_percentage']
            
        # unsupervised image segmentation
        all_segmentations = []
        all_result = []
        all_color_avg = []
        # init image tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        # segmentation ML
        seg_map = skimage.segmentation.felzenszwalb(image, scale=self.config['localization']['felzenszwalb']['scale'], sigma=self.config['localization']['felzenszwalb']['sigma'], min_size=self.config['localization']['felzenszwalb']['min_size'])
        seg_map = seg_map.flatten()
        seg_lab = [np.where(seg_map == u_label)[0]
                   for u_label in np.unique(seg_map)]
        for i in range(self.config['localization']['number_seg_runs']):
            result, color_avg = UnsegRun(self.config['localization']['model'], image, image_tensor, seg_lab, self.device)
            segmentations = []
            for color_id in color_avg:
                tmp = np.zeros_like(result)
                mask = (result == color_id)
                tmp[mask] = 1
                tmp=tmp[...,0]
                segmentations.append(tmp)
            all_segmentations.append(segmentations)
            all_result.append(result)
            all_color_avg.append(color_avg)

        un_seg_mask, un_seg_max_prob_mask, to_out_index = Filter_UnSegMasks(anomaly_map, all_segmentations, prob_diff=self.config['localization']['prob_diff'], max_area_percentage=self.config['localization']['max_area_percentage'], min_area_percentage=self.config['localization']['min_area_percentage'], prob_high_confidence=self.config['localization']['prob_high_confidence'])

        # combine map mask
        mask = combine_mask(self.config, image, anomaly_map, un_seg_mask, un_seg_max_prob_mask)


        result={}
        result['image'] = image
        result['anomaly_map'] = anomaly_map
        result['pred_label'] = pred_label
        result['mask'] = mask
        return result
        
    def localization_(self, image_path=None, image=None, anomaly_map=None, pred_label=True, remove_background=False):
        image = preprocess_image(image_path, image, image_size=self.config['fastflow']['output_size'])

        if anomaly_map is None:
            image_tensor = self.image_transform(image)
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(dim=0).to(self.device)
                outputs = self.model(image_tensor)
            anomaly_map = outputs["anomaly_map"].cpu().detach().numpy()[0,0]
            anomaly_map = normalize_map(anomaly_map, self.config['fastflow']['resume_and_normalize_config']['pixel_threshold'], self.config['fastflow']['resume_and_normalize_config']['pixel_min'], self.config['fastflow']['resume_and_normalize_config']['pixel_max'])
        
        if pred_label is None:
            pred_mask = anomaly_map > 0.5
            area = np.sum(pred_mask > 0)
            prob = area / pred_mask.shape[0]**2
            pred_label = prob > self.config['detection']['abnormal_pixel_percentage']
            
        if remove_background is None:
            remove_background = self.config['background']['remove_background']
            
        # detect with background
        if pred_label and remove_background:
            background = auto_detect_background(image)
            background = skimage.morphology.remove_small_objects(
                background > 0,
                min_size = 400,
                connectivity=1
            ).astype(np.uint8) * 255
            background_score = background_score_func(background, anomaly_map)
            # high background score: background_label_inv is False, no localization is performed
            background_label_inv = (background_score < self.config['background']['bg_ratio_threshold'])
            background_percentage = np.sum(background>0) / background.size
        else:
            # background is not excluded, locate all artifacts
            background_label_inv = True
            background_percentage = 0
            background_score = 0
            background = np.zeros_like(image)
        
        if pred_label and background_label_inv:
            # unsupervised image segmentation
            all_segmentations = []
            all_result = []
            all_color_avg = []
            # init image tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            # segmentation ML
            seg_map = skimage.segmentation.felzenszwalb(image, scale=self.config['localization']['felzenszwalb']['scale'], sigma=self.config['localization']['felzenszwalb']['sigma'], min_size=self.config['localization']['felzenszwalb']['min_size'])
            seg_map = seg_map.flatten()
            seg_lab = [np.where(seg_map == u_label)[0]
                       for u_label in np.unique(seg_map)]
            for i in range(self.config['localization']['number_seg_runs']):
                result, color_avg = UnsegRun(self.config['localization']['model'], image, image_tensor, seg_lab, self.device)
                segmentations = []
                for color_id in color_avg:
                    tmp = np.zeros_like(result)
                    mask = (result == color_id)
                    tmp[mask] = 1
                    tmp=tmp[...,0]
                    segmentations.append(tmp)
                all_segmentations.append(segmentations)
                all_result.append(result)
                all_color_avg.append(color_avg)

            un_seg_mask, un_seg_max_prob_mask, to_out_index = Filter_UnSegMasks(anomaly_map, all_segmentations, prob_diff=self.config['localization']['prob_diff'], max_area_percentage=self.config['localization']['max_area_percentage'], min_area_percentage=self.config['localization']['min_area_percentage'], prob_high_confidence=self.config['localization']['prob_high_confidence'])
            
            # combine map mask
            mask = combine_mask(self.config, image, anomaly_map, un_seg_mask, un_seg_max_prob_mask)
        else:
            mask = np.zeros_like(anomaly_map)


        result={}
        result['image'] = image
        result['anomaly_map'] = anomaly_map
        result['pred_label'] = pred_label
        result['mask'] = mask
        return result
    


