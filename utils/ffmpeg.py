import subprocess
import json

def get_fps(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path
        ],
        capture_output=True,
        text=True,
        check=True
    )
    data = json.loads(result.stdout)
    num, den = data["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)

def convert_to_25fps(input_video, output_video):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", "fps=25",
            "-movflags", "+faststart",
            output_video
        ],
        check=True
    )
