import numpy as np
import argparse
import os
import json
import cv2
import glob
from collections import defaultdict
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
from multiprocessing import Pool, cpu_count
import tqdm 

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

def load_depth(depth_path, is_float16=True):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image==65504)
    else:
        invalid_mask=None
    image = image / 10000 # cm -> 100m
    return image, invalid_mask



def process_scene(args):
    type_, cur_cat, split, scene, save_path, input_dir = args
    json_path = f'{input_dir}/{cur_cat}/{type_}/{split}/{scene}/transforms.json'
    save_path_iter = os.path.join(save_path, cur_cat, type_, split, scene)
    with open(json_path, "r") as f:
        meta = json.load(f)
    
    cam_x = meta['camera_angle_x']
    frames = meta["frames"]

    if type_ == 'aerial':
        w = float(1920)
        h = float(1080)
    else:
        w = float(1000)
        h = float(1000)
    
    fl_x = float(.5 * w / np.tan(.5 * cam_x))
    fl_y = fl_x
    
    cx = w / 2 
    cy = h / 2 
    w = int(w )
    h = int(h )
    fx = fl_x 
    fy = fl_y 
    
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    meta_save = defaultdict(dict)

    for i, frame in tqdm.tqdm(enumerate(frames)):
        file_path = frame['frame_index']
        c2w = np.array(frame["rot_mat"])
        c2w[3, 3] = 1
        c2w = c2w @ OPENGL_TO_OPENCV
        try:
            image_path = os.path.join(f'{input_dir}/{cur_cat}/{type_}/{split}/{scene}/{scene}/', '{:04d}.png'.format(file_path))
            if type_ == 'aerial':
                depth_path = os.path.join(f'{input_dir}/{cur_cat}_depth_float32/{type_}/{split}/{scene}_depth/{scene}_depth/', '{:04d}.exr'.format(file_path))
            else:
                depth_path = os.path.join(f'{input_dir}/{cur_cat}_depth/{type_}/{split}/{scene}_depth/{scene}_depth/', '{:04d}.exr'.format(file_path))
        except:
            import ipdb; ipdb.set_trace()
        rgb = cv2.imread(image_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0] / 10000.  # cm -> 100m
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        
        save_path_iter_frame = os.path.join(save_path_iter, os.path.basename(image_path))
        os.makedirs(os.path.dirname(save_path_iter_frame), exist_ok=True)
        cv2.imwrite(save_path_iter_frame, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        save_path_iter_frame_depth = os.path.join(save_path_iter, os.path.basename(image_path).replace('png', 'exr'))
        cv2.imwrite(save_path_iter_frame_depth, depth.astype(np.float32))
        
        meta_save[(cur_cat, type_, split, scene)][os.path.basename(image_path)] = {"intrinsic": intrinsic, "c2w": c2w}
    
    return meta_save

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess matrixcity")
    parser.add_argument("--input_dir",type=str,
                        default='/home/storage-bucket/haian/zipmap_data/processed_matrixcity/matrixcity_raw'
                        )

    parser.add_argument("--output_dir",type=str,
                        default='/home/storage-bucket/haian/zipmap_data/processed_matrixcity/matrixcity_processed'
                        )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    save_path = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    
    meta_save = defaultdict(dict)
    categories = ['big_city', "small_city"] 
    split_list = ['train', 'test']
    type_list = ['aerial', 'street'] 
    tasks = []
    for cur_cat in categories:
        for cur_type in type_list:
            for cur_split in split_list:
                scenes = glob.glob(f'{input_dir}/{cur_cat}/{cur_type}/{cur_split}/*')
                print(f'find {len(scenes)} scenes under {input_dir}/{cur_cat}/{cur_type}/{cur_split}/*')
                for scene in scenes:
                    scene_name = scene.split('/')[-1]
                    tasks.append((cur_type, cur_cat, cur_split, scene_name, save_path, input_dir))
    
    with Pool(processes=64) as pool:
        results = pool.map(process_scene, tasks)
    final_meta_save = {}
    for result in results:
        final_meta_save.update(result)
    
    # save meta in a single compressed npz
    np.savez_compressed(os.path.join(save_path, 'transforms_all.npz'), final_meta_save)
    print(f'save transforms_all.npz to {save_path}')




                    