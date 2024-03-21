"""
Record RGB+D Image Sequences and Point Clouds using Intel Realsense D435i camera
Creator: Felicia Luo
Date: 3/20/2024
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import json
import open3d as o3d
from os.path import exists, join, abspath
import os
from datetime import datetime

#### SPECIFY THESE ####
# downsample videos
FPS = 3
sec_clip_length = 1
sec_spacing = 0
# Directory to save dataset in
DATASET_FOLDER = '../dataset/'

# calc millisecond intervals per FPS
ms_milestones = [i * 1000 // (FPS) for i in range(FPS)]
ms_milestones.append(1000) # [0, 200, 400, 600, 800, 1000] for FPS = 5

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
align_to = rs.stream.color
align = rs.align(align_to)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
found_rgb = False
if depth_sensor: print("Depth Camera Found")
else: print("RGB Camera NOT Found")

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        print("RGB Camera Found")
        break
if not found_rgb:
    print("RGB Camera NOT Found")
    sys.exit()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
print("** Start streaming **")


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

if __name__ == "__main__":
    # Directory to save dataset in
    startTime = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    OUTPUT_FOLDER = DATASET_FOLDER + str(startTime) + '_fps%d_clip_%d_%d/'%(FPS, sec_clip_length, sec_spacing) # eg. ../dataset/20240320_180546_fps3_clip_1_0/
    PATH_DEPTH  = OUTPUT_FOLDER + 'depth'
    PATH_COLOR = OUTPUT_FOLDER + 'color'
    PATH_PCD_PLY = OUTPUT_FOLDER + 'pcd_ply'
    PATH_PCD_NPZ = OUTPUT_FOLDER + 'pcd_npz'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PATH_DEPTH, exist_ok=True)
    os.makedirs(PATH_COLOR, exist_ok=True)
    os.makedirs(PATH_PCD_PLY, exist_ok=True)
    os.makedirs(PATH_PCD_NPZ, exist_ok=True)

    # Streaming loop
    frame_count = 0
    frame_ms_ind = 0
    prev_sec = -1
    try:
        while True:
            # start from second 00
            if datetime.now().second != 0: continue

            while True:
                dt0 = datetime.now()
                
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # continue waiting if current sec and ms should not be saved
                curr_time = datetime.now().strftime("%H_%M_%S_%f")[:-3] # eg. 16_32_08_325
                curr_sec, curr_ms = int(curr_time[6:8]), int(curr_time[-3:])
                if curr_sec % (sec_clip_length + sec_spacing) >= sec_clip_length: continue
                if curr_ms < ms_milestones[frame_ms_ind] or curr_ms > ms_milestones[frame_ms_ind+1]: continue
                if frame_ms_ind == 0 and curr_sec == prev_sec: continue

                # Align depth and color frame
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Save intrinsic at the beginning
                if frame_count == 0:
                    save_intrinsic_as_json(
                        join(OUTPUT_FOLDER, "camera_intrinsic.json"),
                        color_frame)
                    
                # Save RGB and depth image frames
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite("%s/%s.png" % \
                        (PATH_DEPTH, curr_time), depth_image)
                cv2.imwrite("%s/%s.jpg" % \
                        (PATH_COLOR, curr_time), color_image)
                
                frame_count += 1
                frame_ms_ind += 1
                prev_sec = curr_sec
                if frame_ms_ind >= FPS: frame_ms_ind = 0

                # Save point clouds
                # Create RGBD
                color =  o3d.geometry.Image(np.array(color_frame.get_data()))
                depth =  o3d.geometry.Image(np.array(depth_frame.get_data()))
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
                # Get intrinsic
                tt = profile.get_stream(rs.stream.depth)
                intr = tt.as_video_stream_profile().get_intrinsics()
                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx,
                                                                intr.fy, intr.ppx, intr.ppy)
                # Create point cloud from rgbd
                pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,pinhole_camera_intrinsic)
                # rotate -90 degree by x-axis
                pointcloud.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                
                # Save point cloud
                o3d.io.write_point_cloud(os.path.join(PATH_PCD_PLY, "%s.ply"%(curr_time)), pointcloud)
                # # Save npz
                # xyz = np.asarray(pointcloud.points)
                # rgb = np.asarray(pointcloud.colors)
                # assert(xyz.shape == rgb.shape)
                # out_npz = np.hstack([xyz, rgb])
                # np.savez(os.path.join(PATH_PCD_NPZ, "%s.npz"%(curr_time)), out_npz)
                
                # Print fps
                process_time = datetime.now() - dt0
                print("FPS: "+str(1/process_time.total_seconds())) # avg 30 fps if not set fps

    finally:
        # Stop streaming
        pipeline.stop()
