#
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import os
#
# # Set up RealSense pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
#
# align_to = rs.stream.color
# align = rs.align(align_to)
#
# pipeline.start(config)
#
# os.makedirs("gesture", exist_ok=True)
#
# recording = False
# seq_data = []
# gesture_label = None
# sequence_count = 1
#
# print("Press 'r' to start/stop recording, 'q' to quit.")
#
# try:
#     while True:
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#
#         color_frame = aligned_frames.get_color_frame()
#         depth_frame = aligned_frames.get_depth_frame()
#         if not color_frame or not depth_frame:
#             continue
#
#         color_image = np.asanyarray(color_frame.get_data())
#         depth_image = np.asanyarray(depth_frame.get_data())
#
#         cv2.imshow('RGB Stream', color_image)
#
#         if recording:
#             seq_data.append((color_image.copy(), depth_image.copy()))
#
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord('r'):
#             if not recording:
#                 recording = True
#                 seq_data = []
#                 gesture_label = "2"
#                     # input("Enter gesture label for this recording: ")
#                 print("Recording started for gesture:", gesture_label)
#             else:
#                 recording = False
#                 sequence_count += 1
#                 np.savez(os.path.join("gesture", f"gesture_{gesture_label}_{sequence_count}.npz"),
#                          rgb=np.array([frame[0] for frame in seq_data]),
#                          depth=np.array([frame[1] for frame in seq_data]),
#                          label=gesture_label)
#                 print("Recording stopped. Sequence saved.")
#         elif key == ord('q'):
#             break
#
# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ---------- Params ----------
OUTPUT_DIR = "14"
FPS = 30
MAX_DEPTH_MM = 4000  # for visualization scaling (≈4m)
# ----------------------------

def depth_to_vis(depth_image, max_depth_mm=MAX_DEPTH_MM):
    """Convert 16-bit depth (mm) to an 8-bit color-mapped visualization."""
    depth_clipped = np.clip(depth_image, 0, max_depth_mm).astype(np.uint16)
    depth_8u = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / max_depth_mm)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

def make_writer(base_path_no_ext, size, is_color=True, fps=FPS):
    """
    Try several FourCCs. Returns (writer, path_used, fourcc_str).
    If MP4/H.264 isn't available in your OpenCV build, it will fall back.
    """
    trials = [
        ("avc1", ".mp4"),  # H.264 (needs FFmpeg-enabled OpenCV)
        ("mp4v", ".mp4"),  # MPEG-4 part 2
        ("XVID", ".avi"),  # AVI fallback
        ("MJPG", ".avi"),
    ]
    for fourcc_str, ext in trials:
        out_path = base_path_no_ext + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, size, isColor=is_color)
        if writer.isOpened():
            print(f"[VideoWriter] Using {fourcc_str} -> {out_path}")
            return writer, out_path, fourcc_str
        else:
            writer.release()
    raise RuntimeError(
        "Could not open a VideoWriter. Install an OpenCV build with FFmpeg/H.264 support "
        "(e.g., opencv-python) or install FFmpeg and ensure OpenCV can use it."
    )

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)

align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

os.makedirs(OUTPUT_DIR, exist_ok=True)

recording = False
seq_data = []
gesture_label = None
sequence_count = 0

rgb_writer = None
depth_writer = None
rgb_video_path = None
depth_video_path = None

print("Press 'r' to start/stop recording, 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())        # uint8 BGR
        depth_image = np.asanyarray(depth_frame.get_data())        # uint16 (mm)

        cv2.imshow('RGB Stream', color_image)

        if recording:
            # Save raw frames into memory for .npz (as before)
            seq_data.append((color_image.copy(), depth_image.copy()))

            # Write to videos
            rgb_writer.write(color_image)
            depth_vis = depth_to_vis(depth_image)
            depth_writer.write(depth_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                # ---- START recording ----
                recording = True
                seq_data = []
                gesture_label = "2"
                base_name = f"gesture_{gesture_label}_{sequence_count}"
                base_no_ext = os.path.join(OUTPUT_DIR, base_name)

                # Create VideoWriters
                h, w = color_image.shape[:2]
                rgb_writer, rgb_video_path, _ = make_writer(base_no_ext + "_rgb", (w, h), is_color=True)
                depth_writer, depth_video_path, _ = make_writer(base_no_ext + "_depth", (w, h), is_color=True)

                print(f"Recording started for gesture: {gesture_label}")
            else:
                # ---- STOP recording ----
                recording = False


                # Save NPZ
                npz_path = os.path.join(OUTPUT_DIR, f"gesture_{gesture_label}_{sequence_count}.npz")
                np.savez(
                    npz_path,
                    rgb=np.array([frame[0] for frame in seq_data], dtype=np.uint8),
                    depth=np.array([frame[1] for frame in seq_data], dtype=np.uint16),
                    label=gesture_label
                )

                # Release writers
                if rgb_writer is not None:
                    rgb_writer.release()
                    rgb_writer = None
                if depth_writer is not None:
                    depth_writer.release()
                    depth_writer = None

                print(f"Recording stopped. Sequence saved:\n  NPZ : {npz_path}\n  RGB : {rgb_video_path}\n  DEP : {depth_video_path}")
                sequence_count += 1

        elif key == ord('q'):
            break

finally:
    # Clean up
    if rgb_writer is not None:
        rgb_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
