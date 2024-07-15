import os
from moviepy.editor import VideoFileClip

def change_fps(input_path, output_path, fps=24):
    # 读取视频文件
    clip = VideoFileClip(input_path)
    # 修改帧率
    clip = clip.set_fps(fps)
    # 保存视频文件
    clip.write_videofile(output_path, codec="libx264")

def process_videos(input_folder, output_folder, fps=24):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):  # 你可以根据需要添加其他视频格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print(f"Processing {input_path} -> {output_path}")
            change_fps(input_path, output_path, fps)

def main(folder1, folder2, output_folder1, output_folder2):
    process_videos(folder1, output_folder1)
    process_videos(folder2, output_folder2)

if __name__ == "__main__":
    # 这里替换成你的文件夹路径
    folder1 ='data/resizedwater'  # 替换为第一个文件夹输出路径
    folder2 = 'data/resizedwine'  # 替换为第二个文件夹输出路径

    output_folder1 = 'data/resizedwater24'
    output_folder2 = 'data/resizedwine24'
    
    main(folder1, folder2, output_folder1, output_folder2)
