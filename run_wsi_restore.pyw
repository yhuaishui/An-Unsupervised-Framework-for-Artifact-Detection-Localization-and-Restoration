import os
import argparse
import numpy as np
from PIL import Image
import torch
import random
from tqdm import tqdm
import time
from multiprocessing import Process, Array, Value, freeze_support
import ctypes
import pandas as pd
import openslide
import pyvips
from pipeline.parser import parse

def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_pipeline.json', help='JSON file for config')
    return parser.parse_args(args_list)

def split_coords(patch_coords, num_processes):
    coords_list = list(patch_coords)
    random.shuffle(coords_list)
    chunk_size = max(1, len(coords_list) // num_processes)
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes-1 else len(coords_list)
        chunks.append(coords_list[start:end])
    return chunks

class SharedWSIMaskWriter:
    def __init__(self, wsi_path, mask_path, restore_output_path, patch_size=(512,512), 
                 restore_shared=None, processed_patches=None):
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.restore_output_path = restore_output_path
        self.patch_size = patch_size
        
        with openslide.OpenSlide(wsi_path) as wsi:
            self.level0_w, self.level0_h = wsi.level_dimensions[0]
        
        if restore_shared is None or processed_patches is None:
            self.restore_shared = Array('B', self.level0_w * self.level0_h * 3, lock=True) 
            self.processed_patches = Value('i', 0, lock=True)
            
            ctypes.memset(
                ctypes.addressof(self.restore_shared.get_obj()),
                0xff,  # 255
                self.level0_w * self.level0_h * 3
            )
            
            self._bind_shared_memory()
        else:
            self.restore_shared = restore_shared
            self.processed_patches = processed_patches
            self._bind_shared_memory()

    def _bind_shared_memory(self):
        self.restore_ctypes = self.restore_shared.get_obj()
        
        self.level0_restore = np.frombuffer(self.restore_ctypes, dtype=np.uint8)
        self.level0_restore = self.level0_restore.reshape((self.level0_h, self.level0_w, 3))

    def read_patch_from_wsi(self, x, y):
        try:
            with openslide.OpenSlide(self.wsi_path) as wsi:
                patch = wsi.read_region((x, y), 0, self.patch_size).convert('RGB')
                return patch
        except Exception as e:
            print(f"read patch failed (x={x}, y={y}): {e}")
            return None
        
    def read_patch_from_mask(self, x, y):
        try:
            with openslide.OpenSlide(self.mask_path) as mask:
                patch = mask.read_region((x, y), 0, self.patch_size).convert('RGB')
                return patch
        except Exception as e:
            print(f"read patch failed (x={x}, y={y}): {e}")
            return None

    def write_img_to_wsi(self, x, y, restore_img):
        x_end = min(x + self.patch_size[0], self.level0_w)
        y_end = min(y + self.patch_size[1], self.level0_h)
        valid_w = x_end - x
        valid_h = y_end - y
        if valid_w <= 0 or valid_h <= 0:
            return

        with self.restore_shared.get_lock():  
            self.level0_restore[y:y_end, x:x_end, :] = restore_img[:valid_h, :valid_w, :]
            self.processed_patches.value += 1
        
        if hasattr(self.restore_ctypes, 'flush'):
            self.restore_ctypes.flush()

    def generate_pyramid_mask(self):
        restore_data = np.ascontiguousarray(self.level0_restore)

        restore_img = pyvips.Image.new_from_memory(
            restore_data.tobytes(),
            self.level0_w, self.level0_h, 3, "uchar"
        )
        restore_img.tiffsave(
            self.restore_output_path,
            pyramid=True, tile=True, tile_width=512, tile_height=512,
            compression="lzw",
            bigtiff=True
        )
        
def worker_task(task_id, wsi_path, mask_path, restore_output_path, patch_size,
                restore_shared, processed_patches, patch_coords, model_config, gpu_id):
    writer = SharedWSIMaskWriter(
        wsi_path=wsi_path,
        mask_path=mask_path,
        restore_output_path=restore_output_path,
        patch_size=patch_size,
        restore_shared=restore_shared,
        processed_patches=processed_patches,
    )
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError(f"Process {task_id} binding to GPU {gpu_id} is unavailable!")
    print(f"Process {task_id} has been bound to GPU {gpu_id} (current device: {torch.cuda.current_device()})")

    from pipeline.artifact_restoration import RestorationModel
    restoration_model = RestorationModel(model_config)

    processed_count = 0
    for x, y in tqdm(patch_coords, desc=f"Process {task_id} processing patches"):
        patch = writer.read_patch_from_wsi(x, y)
        mask = writer.read_patch_from_mask(x, y)
        if patch is None:
            continue
        
        restore_img = restoration_model.inpaint(image=patch, mask=mask, progress=False)
        restore_img = np.array(Image.fromarray(restore_img['restore']).resize(patch_size, Image.BICUBIC))
        
        writer.write_img_to_wsi(x, y, restore_img)
        processed_count += 1

    print(f"Process {task_id} finished: processed {processed_count} patches")
    return task_id


def main(wsi_path, mask_path, restore_output_path, normal_coords, abnormal_coords, abnormal_coords_2, model_config, patch_size=(512,512), num_processes=1, gpu_ids=[0]):
    writer = SharedWSIMaskWriter(
        wsi_path=wsi_path,
        mask_path=mask_path,
        restore_output_path=restore_output_path,
        patch_size=patch_size
    )
    
    processed_count = 0
    for x, y in tqdm(normal_coords, desc=f"Process normal patches"):
        patch = writer.read_patch_from_wsi(x, y)
        patch = np.array(patch)
        if patch is None:
            continue
        writer.write_img_to_wsi(x, y, patch)
        processed_count += 1
        
    '''Process patches with an excessively large mask; fill it with white'''
    # if len(abnormal_coords_2) != 0:
    #     for x, y in tqdm(abnormal_coords_2, desc=f"Process patches with an excessively large mask"):
    #         patch = img = np.full((512, 512, 3), 255, dtype=np.uint8)
    #         writer.write_img_to_wsi(x, y, patch)
    #         processed_count += 1
            
    coord_chunks = split_coords(abnormal_coords, num_processes)
    processes = []
    for i in range(num_processes):
        if len(coord_chunks[i]) == 0:
            continue
        p = Process(
            target=worker_task,
            args=(
                i, wsi_path, mask_path, restore_output_path, patch_size,
                writer.restore_shared, writer.processed_patches, coord_chunks[i], 
                model_config, gpu_ids[i]
            )
        )
        processes.append(p)
        p.start()
        print(f"Launching process {i} (PID: {p.pid}) to process {len(coord_chunks[i])} patches")
    
    for p in processes:
        p.join()  
        if p.exitcode == 0:
            print(f"Process {p.pid} completed normally")
        else:
            print(f"Process {p.pid} exited abnormally with exit code: {p.exitcode}")

    if hasattr(writer.restore_ctypes, 'flush'):
        writer.restore_ctypes.flush()

    print(f"\n=== Final Verification ===")
    print(f"Total patches processed: {writer.processed_patches.value}")

    if writer.processed_patches.value > 0:
        writer.generate_pyramid_mask()
        print(f"Successfully generated TIFF file：{restore_output_path}")
    else:
        print("❌ No valid data, skipping TIFF generation")


if __name__ == "__main__":
    start = time.time()
    
    args = parse_args(['--config', './config/config_pipeline.json'])
    config = parse(args.config)

    
    WSI_PATH = './data/sample_wsi_roi/img.tiff'
    MASK_PATH = './data/sample_wsi_roi/mask.tiff'
    CSV_PATH = './data/sample_wsi_roi/detail_result.csv'
    RESTORE_OUTPUT_PATH = './data/sample_wsi_roi/restore.tiff'
    PATCH_SIZE = (512, 512)
    
    df = pd.read_csv(CSV_PATH)
    normal_coords = df[df["refined_artifact_label"] == False]
    normal_coords = list(zip(normal_coords["x"], normal_coords["y"]))
    
    abnormal_coords = df[(df["refined_artifact_label"] == True) & (df["mask_percentage"] < 0.7)]
    abnormal_coords = list(zip(abnormal_coords["x"], abnormal_coords["y"]))
    
    # The mask is too large; fill it with white.
    # abnormal_coords_2 = df[(df["refined_artifact_label"] == True) & (df["mask_percentage"] >= 0.7)]
    # abnormal_coords_2 = list(zip(abnormal_coords_2["x"], abnormal_coords_2["y"]))

    main(
        wsi_path=WSI_PATH,
        mask_path=MASK_PATH,
        restore_output_path=RESTORE_OUTPUT_PATH,
        normal_coords=normal_coords,
        abnormal_coords=abnormal_coords,
        abnormal_coords_2=None,
        model_config=config,
        patch_size=PATCH_SIZE,
        num_processes=1,
        gpu_ids = [0]
    )

    end = time.time()
    print(f'Total runtime: {end-start:.5f} sec')