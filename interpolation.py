import argparse
from pathlib import Path
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from config import DATA_DIR, RESULTS_DIR


def interpolate_and_render(pose_path: str, output_video: str, target_fps: int = 24):
    """Interpolates pose frames to a target FPS and saves the rendered video."""
    pose_file = Path(pose_path)
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found at: {pose_file}")

    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())
        pose.interpolate_fps(target_fps, kind='linear')

    visualizer = PoseVisualizer(pose)
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    visualizer.save_video(output_video, visualizer.draw())
    print(f"✅ Interpolated video successfully saved to: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate Pose and Render Video")
    parser.add_argument(
        "--pose",
        type=str,
        default=str(DATA_DIR / "sample_poses" / "chao.pose"),
        help="Path to the input .pose file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "interpolation_test.mp4"),
        help="Path to the output .mp4 video"
    )
    parser.add_argument("--fps", type=int, default=24, help="Target frame rate")

    args = parser.parse_args()
    interpolate_and_render(args.pose, args.output, args.fps)
