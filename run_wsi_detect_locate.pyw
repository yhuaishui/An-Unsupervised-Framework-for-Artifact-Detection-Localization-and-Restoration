import os
import torch
import sys
import glob
import numpy as np
import openslide
import pyvips
from tqdm import tqdm
import h5py
import cv2
import time
import random
from multiprocessing import Process, Array, Value, freeze_support
import ctypes
import pandas as pd

class PatchInfo(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),     # x-coordinate
        ("y", ctypes.c_int),     # y-coordinate
        ("artifact_label", ctypes.c_bool), # artifact label based on anomaly map
        ("background_score", ctypes.c_float), # overlap score between background and anomaly map
        ("background_percentage", ctypes.c_float),   # background pixel ratio within the patch
        ("is_background", ctypes.c_bool),  # whether patch is background (based on overlap with anomaly map)
        ("refined_artifact_label", ctypes.c_bool),  # refined artifact label after mask cleaning
        ("mask_percentage", ctypes.c_float), # mask pixel ratio within the patch
    ]

class SharedWSIMaskWriter:
    def __init__(self, wsi_path, mask_output_path, map_output_path, patch_size=(512,512), 
                 mask_shared=None, map_shared=None, processed_patches=None, len_patch_coords=0, patch_coords_results_shared=None):
        self.wsi_path = wsi_path
        self.mask_output_path = mask_output_path
        self.map_output_path = map_output_path
        self.patch_size = patch_size
        self.len_patch_coords = len_patch_coords
        
        with openslide.OpenSlide(wsi_path) as wsi:
            self.level0_w, self.level0_h = wsi.level_dimensions[0]
        
        if mask_shared is None or map_shared is None or processed_patches is None or patch_coords_results_shared is None:
            self.mask_shared = Array('B', self.level0_w * self.level0_h, lock=True)  
            self.map_shared = Array('B', self.level0_w * self.level0_h, lock=True)
            self.processed_patches = Value('i', 0, lock=True) 
            self.patch_coords_results_shared = Array(PatchInfo, self.len_patch_coords, lock=True)
            self._bind_shared_memory()
        else:
            self.mask_shared = mask_shared
            self.map_shared = map_shared
            self.processed_patches = processed_patches
            self.patch_coords_results_shared = patch_coords_results_shared
            self._bind_shared_memory()

    def _bind_shared_memory(self):
        self.mask_ctypes = self.mask_shared.get_obj()
        self.map_ctypes = self.map_shared.get_obj()
        
        self.level0_mask = np.frombuffer(self.mask_ctypes, dtype=np.uint8)
        self.level0_mask = self.level0_mask.reshape((self.level0_h, self.level0_w))
        
        self.level0_map = np.frombuffer(self.map_ctypes, dtype=np.uint8)
        self.level0_map = self.level0_map.reshape((self.level0_h, self.level0_w))

    def read_patch_from_wsi(self, x, y):
        try:
            with openslide.OpenSlide(self.wsi_path) as wsi:
                patch = wsi.read_region((x, y), 0, self.patch_size).convert('RGB')
                return np.array(patch)
        except Exception as e:
            print(f"read patch failed (x={x}, y={y}): {e}")
            return None

    def write_mask_to_wsi(self, x, y, mask, anomaly_map, artifact_label, background_score, background_percentage, is_background, refined_artifact_label, mask_percentage):
        x_end = min(x + self.patch_size[0], self.level0_w)
        y_end = min(y + self.patch_size[1], self.level0_h)
        valid_w = x_end - x
        valid_h = y_end - y
        if valid_w <= 0 or valid_h <= 0:
            return

        with self.mask_shared.get_lock():  
            self.level0_mask[y:y_end, x:x_end] = mask[:valid_h, :valid_w]
            self.level0_map[y:y_end, x:x_end] = anomaly_map[:valid_h, :valid_w]
            # 
            self.patch_coords_results_shared[self.processed_patches.value].x = x
            self.patch_coords_results_shared[self.processed_patches.value].y = y
            self.patch_coords_results_shared[self.processed_patches.value].artifact_label = artifact_label
            self.patch_coords_results_shared[self.processed_patches.value].background_score = background_score
            self.patch_coords_results_shared[self.processed_patches.value].background_percentage = background_percentage
            self.patch_coords_results_shared[self.processed_patches.value].is_background = is_background
            self.patch_coords_results_shared[self.processed_patches.value].refined_artifact_label = refined_artifact_label
            self.patch_coords_results_shared[self.processed_patches.value].mask_percentage = mask_percentage
            self.processed_patches.value += 1
        
        if hasattr(self.mask_ctypes, 'flush'):
            self.mask_ctypes.flush()
        if hasattr(self.map_ctypes, 'flush'):
            self.map_ctypes.flush()

    def generate_pyramid_mask(self):
        mask_data = np.ascontiguousarray(self.level0_mask)
        map_data = np.ascontiguousarray(self.level0_map)

        mask_img = pyvips.Image.new_from_memory(
            mask_data.tobytes(),
            self.level0_w, self.level0_h, 1, "uchar"
        )
        mask_img.tiffsave(
            self.mask_output_path,
            pyramid=True, tile=True, tile_width=512, tile_height=512,
            compression="lzw"#, bigtiff=True, subifd=True
        )

        map_img = pyvips.Image.new_from_memory(
            map_data.tobytes(),
            self.level0_w, self.level0_h, 1, "uchar"
        )
        map_img.tiffsave(
            self.map_output_path,
            pyramid=True, tile=True, tile_width=512, tile_height=512,
            compression="lzw"#, bigtiff=True, subifd=True
        )

def worker_task(task_id, wsi_path, mask_output_path, map_output_path, patch_size,
                mask_shared, map_shared, processed_patches, patch_coords, len_patch_coords, patch_coords_results_shared,
                config_path, gpu_id):

    writer = SharedWSIMaskWriter(
        wsi_path=wsi_path,
        mask_output_path=mask_output_path,
        map_output_path=map_output_path,
        patch_size=patch_size,
        mask_shared=mask_shared,
        map_shared=map_shared,
        processed_patches=processed_patches,
        len_patch_coords=len_patch_coords,
        patch_coords_results_shared=patch_coords_results_shared
    )
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError(f"Process {task_id} binding to GPU {gpu_id} is unavailable!")
    print(f"Process {task_id} has been bound to GPU {gpu_id} (current device: {torch.cuda.current_device()})")

    from pipeline.artifact_detection_localization import HisAnomalyModel
    HisAnomaly = HisAnomalyModel(
        config = config_path
    )

    processed_count = 0
    for x, y in tqdm(patch_coords, desc=f"Process {task_id} processing patches"):
        patch = writer.read_patch_from_wsi(x, y)
        if patch is None:
            continue
        
        result = HisAnomaly.detection_and_localization(image=patch, detail=True)
        mask = (result['mask']*255).astype(np.uint8)
        mask = cv2.resize(mask, patch_size, interpolation=cv2.INTER_CUBIC)
        anomaly_map = (result['anomaly_map']*255).astype(np.uint8)
        anomaly_map = cv2.resize(anomaly_map, patch_size, interpolation=cv2.INTER_CUBIC)
        
        writer.write_mask_to_wsi(x, y, mask, anomaly_map, result['pred_label'], result['background_score'], result['background_percentage'], result['background_label'], result['refine_pred_label'], result['mask_percentage'])
        processed_count += 1

    print(f"Process {task_id} finished: processed {processed_count} patches")
    return task_id

def main(wsi_path, mask_output_path, map_output_path, CSV_OUTPUT_PATH, patch_coords, patch_size=(512,512), num_processes=1, gpu_ids=[0], config_path='config/config_detection_localization.json'):
    writer = SharedWSIMaskWriter(
        wsi_path=wsi_path,
        mask_output_path=mask_output_path,
        map_output_path=map_output_path,
        patch_size=patch_size,
        len_patch_coords=len(patch_coords)
    )

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
    coord_chunks = split_coords(patch_coords, num_processes)

    processes = []
    for i in range(num_processes):
        if len(coord_chunks[i]) == 0:
            continue
        p = Process(
            target=worker_task,
            args=(
                i, wsi_path, mask_output_path, map_output_path, patch_size,
                writer.mask_shared, writer.map_shared, writer.processed_patches, coord_chunks[i], writer.len_patch_coords, writer.patch_coords_results_shared,
                config_path, gpu_ids[i]
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

    if hasattr(writer.mask_ctypes, 'flush'):
        writer.mask_ctypes.flush()
    if hasattr(writer.map_ctypes, 'flush'):
        writer.map_ctypes.flush()

    print(f"\n=== Final Verification ===")
    print(f"Total patches processed: {writer.processed_patches.value}")
    print(f"Mask non-zero pixels: {np.count_nonzero(writer.level0_mask)}")
    print(f"Map non-zero pixels: {np.count_nonzero(writer.level0_map)}")
    print(f"Mask shape: {writer.level0_mask.shape}")

    if writer.processed_patches.value > 0 and np.count_nonzero(writer.level0_mask) > 0:
        writer.generate_pyramid_mask()
        print(f"Successfully generated TIFF file: {mask_output_path}")
    else:
        print("❌ No valid data, skipping TIFF generation")
        
    csv_data = [ [item.x, item.y, item.artifact_label, item.background_score, item.background_percentage, item.is_background, item.refined_artifact_label, item.mask_percentage] for item in writer.patch_coords_results_shared ]
    df = pd.DataFrame(csv_data, columns=["x", "y", "artifact_label", "background_score", "background_percentage", "is_background", "refined_artifact_label", "mask_percentage"])
    df["xy_tuple"] = list(zip(df["x"], df["y"])) 
    df = df.set_index("xy_tuple").loc[patch_coords].reset_index() 
    df = df.drop(columns=["xy_tuple"]) 
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")
        
if __name__ == "__main__":
    
    start = time.time()

    WSI_PATH = './data/sample_wsi_roi/img.tiff'
    MASK_OUTPUT_PATH = './data/sample_wsi_roi/mask.tiff'
    MAP_OUTPUT_PATH = './data/sample_wsi_roi/map.tiff'
    CSV_OUTPUT_PATH = './data/sample_wsi_roi/detail_result.csv'
    PATCH_SIZE = (512, 512)

    coord_path = './data/sample_wsi_roi/coords.h5'
    with h5py.File(coord_path,'r') as f:
        patch_coords = [(x, y) for x, y in f['coords']] 

    main(
        wsi_path=WSI_PATH,
        mask_output_path=MASK_OUTPUT_PATH,
        map_output_path=MAP_OUTPUT_PATH,
        CSV_OUTPUT_PATH=CSV_OUTPUT_PATH,
        patch_coords=patch_coords,
        patch_size=PATCH_SIZE,
        num_processes=1,
        config_path='config/config_detection_localization.json',
        gpu_ids = [0]
    )

    end = time.time()
    print(f'Total runtime: {end-start:.3f} sec')
