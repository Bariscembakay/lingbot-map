#!/usr/bin/env python3
#
# --------------------------------------------------------
# Preprocessing code for the BlendedMVS dataset based on dust3r
# dataset at https://github.com/YoYo000/BlendedMVS
# 1) Download BlendedMVS.zip
# 2) Download BlendedMVS+.zip
# 3) Download BlendedMVS++.zip
# 4) Unzip everything in the same /path/to/tmp/blendedMVS/ directory
# 5) python datasets_preprocess/preprocess_blendedMVS.py --blendedmvs_dir /path/to/tmp/blendedMVS/
# --------------------------------------------------------
# Compared with the original dust3r_processing.py, this version does resizing the images

import os
import os.path as osp
import re
from tqdm import tqdm
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


def starcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(*args)


def starstarcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(**args)

def parallel_threads(
    function,
    args,
    workers=0,
    star_args=False,
    kw_args=False,
    front_num=1,
    Pool=ThreadPool,
    **tqdm_kw
):
    """tqdm but with parallel execution.

    Will essentially return
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Note:
        the <front_num> first elements of args will not be parallelized.
        This can be useful for debugging.
    """
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float("inf")

    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(
            function(*a) if star_args else function(**a) if kw_args else function(a)
        )

    out = []
    with Pool(workers) as pool:

        if star_args:
            futures = pool.imap(starcall, [(function, a) for a in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, a) for a in args])
        else:
            futures = pool.imap(function, args)

        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--blendedmvs_dir', required=True)
    parser.add_argument('--output_dir', default='data/blendedmvs_processed')
    return parser


def main(db_root, output_dir):
    print('>> Listing all sequences')
    sequences = [f for f in os.listdir(db_root) if len(f) == 24]
    # should find 502 scenes
    assert sequences, f'did not found any sequences at {db_root}'
    print(f'   (found {len(sequences)} sequences)')

    for i, seq in enumerate(tqdm(sequences)):
        out_dir = osp.join(output_dir, seq)
        os.makedirs(out_dir, exist_ok=True)
        # generate the crops
        root = osp.join(db_root, seq)
        cam_dir = osp.join(root, 'cams')
        func_args = [(root, f[:-8], out_dir) for f in os.listdir(cam_dir) if not f.startswith('pair')]
        parallel_threads(load_crop_and_save, func_args, star_args=True, leave=False)

    # # verify that all pairs are there
    # pairs = np.load(pairs_path)
    # for seqh, seql, img1, img2, score in tqdm(pairs):
    #     for view_index in [img1, img2]:
    #         impath = osp.join(output_dir, f"{seqh:08x}{seql:016x}", f"{view_index:08n}.jpg")
    #         assert osp.isfile(impath), f'missing image at {impath=}'

    # print(f'>> Done, saved everything in {output_dir}/')


def load_crop_and_save(root, img, out_dir):
    if osp.isfile(osp.join(out_dir, img + '.npz')):
        return  # already done

    # load everything
    intrinsics_in, R_camin2world, t_camin2world = _load_pose(osp.join(root, 'cams', img + '_cam.txt'))
    # color_image_in = cv2.cvtColor(cv2.imread(osp.join(root, 'blended_images', img +
    #                               '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depthmap_in = load_pfm_file(osp.join(root, 'rendered_depth_maps', img + '.pfm'))

    # do the crop
    H, W = depthmap_in.shape[:2]
    assert H * 4 == W * 3

    depthmap, intrinsics_out =  depthmap_in, intrinsics_in
    R_in2out = np.eye(3)
    # write everything
    # image.save(osp.join(out_dir, img + '.jpg'), quality=80)
    output_image_path = osp.join(out_dir, img + '.jpg')
    origin_image_path = osp.join(root, 'blended_images', img + '.jpg')
    # copy image
    if not osp.isfile(output_image_path):
        os.system(f'cp {origin_image_path} {output_image_path}')
    cv2.imwrite(osp.join(out_dir, img + '.exr'), depthmap)

    # New camera parameters
    R_camout2world = R_camin2world @ R_in2out.T
    t_camout2world = t_camin2world
    np.savez(osp.join(out_dir, img + '.npz'), intrinsics=intrinsics_out,
             R_cam2world=R_camout2world, t_cam2world=t_camout2world)




def _load_pose(path, ret_44=False):
    f = open(path)
    RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
    assert RT.shape == (4, 4)
    RT = np.linalg.inv(RT)  # world2cam to cam2world

    K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
    assert K.shape == (3, 3)

    if ret_44:
        return K, RT
    return K, RT[:3, :3], RT[:3, 3]  # , depth_uint8_to_f32


def load_pfm_file(file_path):
    with open(file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').strip()

        if header == 'PF':
            is_color = True
        elif header == 'Pf':
            is_color = False
        else:
            raise ValueError('The provided file is not a valid PFM file.')

        dimensions = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError('Invalid PFM header format.')

        endian_scale = float(file.readline().decode('UTF-8').strip())
        if endian_scale < 0:
            dtype = '<f'  # little-endian
        else:
            dtype = '>f'  # big-endian

        data_buffer = file.read()
        img_data = np.frombuffer(data_buffer, dtype=dtype)

        if is_color:
            img_data = np.reshape(img_data, (img_height, img_width, 3))
        else:
            img_data = np.reshape(img_data, (img_height, img_width))

        img_data = cv2.flip(img_data, 0)

    return img_data


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.blendedmvs_dir, args.output_dir)