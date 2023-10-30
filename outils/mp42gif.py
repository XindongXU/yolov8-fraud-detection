from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.fx import resize

def speed_up_video(input_file, output_file, speed_factor):
    video_clip = VideoFileClip(input_file)
    accelerated_clip = video_clip.speedx(speed_factor)
    accelerated_clip.write_videofile(output_file)

def video_to_gif(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    target_resolution = (video_clip.size[0] // 4.2, video_clip.size[1] // 4.2)
    resized_clip = video_clip.resize(target_resolution)
    frames = [frame for frame in resized_clip.iter_frames()]
    gif_clip = ImageSequenceClip(frames, fps = 12)
    gif_clip.write_gif(output_file)

input_file_path = 'demo_20230721163319.mp4'
output_file_path = 'accelerated.mp4'
speed_factor = 3  # 加速两倍
speed_up_video(input_file_path, output_file_path, speed_factor)

input_file_path = 'accelerated.mp4'
output_file_path = 'output.gif'
video_to_gif(input_file_path, output_file_path)