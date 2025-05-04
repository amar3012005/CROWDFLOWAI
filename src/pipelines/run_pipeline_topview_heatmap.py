import subprocess
import glob
import os

VIDEO_PATH = "video2.mp4"  # Input video
MODEL_PATH = "yolov8x.pt"
CHECKPOINTS_DIR = "checkpoints"
GIF_OUTPUT = "topview_tracks_heatmap.gif"
GIF_FPS = 30.0  # Match input video FPS
GIF_DELAY = 1.0 / GIF_FPS

# 1. Run YOLO+DeepSORT
print("Running detection and tracking...")
subprocess.run([
    "python", "run_yolov8x.py",
    "--video", VIDEO_PATH,
    "--model", MODEL_PATH,
    "--checkpoints_dir", CHECKPOINTS_DIR
], check=True)

# 2. Find latest parquet
parquets = sorted(glob.glob("trajectories/*.parquet"), key=os.path.getmtime, reverse=True)
if not parquets:
    raise RuntimeError("No parquet files found in trajectories/")
parquet_file = parquets[0]

# 3. Animate topview tracks with heatmap
print(f"Animating topview tracks with heatmap from {parquet_file} at {GIF_FPS} FPS...")
subprocess.run([
    "python", "src/animate_topview_tracks_with_heatmap.py",
    "--traj_file", parquet_file,
    "--output_gif", GIF_OUTPUT,
    "--delay", str(GIF_DELAY),
    "--sigma", "10"
], check=True)

print(f"Pipeline complete. Output: {GIF_OUTPUT}") 