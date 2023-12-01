"""
Export Poind Cloud from Intel Realsense D435i camera
Creator: Felicia Luo, Nov 1st, 2023
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import open3d as o3d
from datetime import datetime

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
pc= rs.pointcloud()
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

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        print("Found Camera")
        break
if not found_rgb:
    print("Camera Not Found")
    sys.exit()

# for Intel Realsense D435i camera
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

out_npz = None



if __name__ == "__main__":
    
    try:

        # # rs save mesh 
        # while True:
        #     dt0=datetime.now()
        #     # Wait for the next set of frames from the camera
        #     frames = pipeline.wait_for_frames()

        #     # Get frameset of color and depth
        #     frames = pipeline.wait_for_frames()
        #     depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()
        #     # print("get depth and color frame")
        #     aligned_frames = align.process(frames)
        #     depth = aligned_frames.get_depth_frame()
        #     color = aligned_frames.get_color_frame()
        #     # print("aligned depth and color frame")
        #     pc.map_to(color)
        #     points = pc.calculate(depth)

        #     # Save pointcloud
        #     print("Saving to 1.ply...")
        #     points.export_to_ply("1.ply", color)
        #     print("Done")

        #     # Print fps
        #     process_time = datetime.now() - dt0
        #     print("FPS: "+str(1/process_time.total_seconds()))


        # open3d fast save pointcloud
        # vis = o3d.visualization.Visualizer()
        # vis.create_window('PCD', width=1280, height=720)
        pointcloud = o3d.geometry.PointCloud()
        
        while True:
            pointcloud.clear()
            # vis.add_geometry(pointcloud)
            dt0=datetime.now()
            print("dt0", dt0.hour, dt0.minute, dt0.second)

            # # only compute and save pointcloud once every 1 seconds
            # if not dt0.second % 1 == 0: continue

            # Wait for the next set of frames from the camera
            frames = pipeline.wait_for_frames()

            # Align depth and color frame
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # print("get depth and color frame")
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            # print("aligned depth and color frame")

            # Create RGBD
            color =  o3d.geometry.Image(np.array(aligned_color_frame.get_data()))
            depth =  o3d.geometry.Image(np.array(aligned_depth_frame.get_data()))
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
            
            '''Save point cloud'''
            filename = 'output/' + str(dt0.hour) + '_' + str(dt0.minute) + '_' + str(dt0.second)
            o3d.io.write_point_cloud(filename+".ply", pointcloud)
            print(filename+".ply saved successfully")

            xyz = np.asarray(pointcloud.points)
            rgb = np.asarray(pointcloud.colors)
            assert(xyz.shape == rgb.shape)
            curr_npz = np.hstack([xyz, rgb])
            # print("curr_npz", curr_npz.shape)
            if out_npz is None: 
                out_npz = curr_npz
                print("save curr")
            else: 
                out_npz = np.vstack([out_npz, curr_npz])
                print('save added')
            np.savez("output/all_pcd.npz", out_npz)
            print(filename + ".npz saved successfully", out_npz.shape)

            # Visualize point cloud
            # vis.update_geometry(pointcloud)
            # vis.poll_events()
            # vis.update_renderer()
            # o3d.visualization.draw_geometries([pointcloud])

            # Print fps
            process_time = datetime.now() - dt0
            print("FPS: "+str(1/process_time.total_seconds()))



    finally:
        # Stop streaming
        pipeline.stop()
