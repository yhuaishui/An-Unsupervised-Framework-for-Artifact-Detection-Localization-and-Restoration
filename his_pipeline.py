import os
import cv2
import argparse
import numpy as np
from pipeline.parser import parse, load_config
from pipeline.artifact_detection_localization import HisAnomalyModel
from pipeline.artifact_restoration import RestorationModel

class Pathological:
    def __init__(self, config_path='config/config_pipeline.json'):
        config = parse(config_path)
        self.config = config
        self.detect_locate_model = HisAnomalyModel(config['pipeline']['detection_and_localization']['config_path'])
        self.restore_model = RestorationModel(config)

    def pipeline(self, image_path=None, image=None, progress=False):
        result = self.detect_locate_model.detection_and_localization(image_path=image_path, image=image)
        if np.sum(result['mask']) == 0:
            result['restore'] = result['image']
        else:
            mask = (result['mask']*255).astype(np.uint8)
            restore_result = self.restore_model.inpaint(image_path=image_path, image=image, mask=mask, progress=progress)
            result['restore'] = restore_result['restore']
        return result
    
    def detect(self, image_path=None, image=None):
        result = self.detect_locate_model.detection(image_path=image_path, image=image)
        return result
    
    def locate(self, image_path=None, image=None, anomaly_map=None):
        result = self.detect_locate_model.localization(image_path=image_path, image=image, anomaly_map=anomaly_map)
        return result
    
    def restore(self, image_path=None, image=None, mask_path=None, mask=None, progress=False):
        result = self.restore_model.inpaint(image_path=image_path, image=image, mask_path=mask_path, mask=mask, progress=False)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=manage_path, default='./config/config_harp.json', help='JSON file for config')
    parser.add_argument('-i', '--image', type=manage_path, default='./sample_data/test.png', help='image file for debug')
    args = parser.parse_args()

    Pipeline = Pathological(args.config)
    result = Pipeline.pipeline(image_path=args.image_path)