import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# 使用Multicore-TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import re

# from models import models_vit
from model.marlin import Marlin
from joblib import Parallel, delayed

# model parameters

device = 'cuda'
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def change_fps(cap, target_fps):
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = frame_count / original_fps
#     new_frame_count = int(duration * target_fps)
#     frames = []

#     for i in range(new_frame_count):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i * original_fps / target_fps)
#         ret, frame = cap.read()
#         if ret:
#             frames.append(frame)
#         else:
#             break

#     return frames

# def face_crop(frame, size):
#     height, width, _ = frame.shape
#     new_width, new_height = size

#     left = int((width - new_width) / 2)
#     top = int((height - new_height) / 2)
#     right = left + new_width
#     bottom = top + new_height

#     cropped_frame = frame[top:bottom, left:right]
#     return cropped_frame

# def crop_video(input_path, output_path, size, target_fps=24):
#     cap = cv2.VideoCapture(input_path)
#     frames = change_fps(cap, target_fps)
#     cap.release()
    
#     if not frames:
#         print("Error: No frames captured.")
#         return

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于编码视频的编码器
#     out = cv2.VideoWriter(output_path, fourcc, target_fps, size)

#     # 在第10帧上进行人脸检测
#     frame = frames[9] if len(frames) > 9 else frames[-1]  # 选择第10帧或最后一帧
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     if len(faces) == 0:
#         print("No faces detected.")
#         return

#     (x, y, w, h) = faces[0]  # 选择检测到的第一个人脸

#     for frame in frames:
#         face_region = frame[y:y+h, x:x+w]
#         resized_face = cv2.resize(face_region, size)
#         out.write(resized_face)

#     out.release()

# def process_videos(input_folder, output_folder, size=(224, 224), target_fps=24):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 获取输入文件夹的名称
#     input_folder_name = os.path.basename(input_folder)

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".mp4") or filename.endswith(".avi"):  # 根据需要可以添加更多的视频格式
#             input_path = os.path.join(input_folder, filename)
#             # 构造新的输出文件名，包含原始文件夹的名称
#             output_filename = f"{input_folder_name}_resized_{filename}"
#             output_path = os.path.join(output_folder, output_filename)
#             crop_video(input_path, output_path, size, target_fps)

# input_folder_1 ='data/test/anger'  # 替换为第一个文件夹输出路径

# input_folder_2 = 'data/test/happy'  # 替换为第二个文件夹输出路径

# output_folder_1 = 'data/test/reanger24'

# output_folder_2 = 'data/test/rehappy24'
# # 处理第一个文件夹
# process_videos(input_folder_1, output_folder_1)

# # 处理第二个文件夹
# process_videos(input_folder_2, output_folder_2)

input_folder_1 ='data/test/reanger24'

input_folder_2 = 'data/test/rehappy24'


# 确保模型在GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Marlin.from_file("marlin_vit_base_ytf", "saved/marlin_vit_base_ytf.encoder.pt").to(device)
print(model)

# 定义提取特征的函数
def extract_features_from_video(video_path):
    features = model.extract_video(video_path)
    mean_features = np.mean(features.squeeze().detach().cpu().numpy(), axis=0)
    return mean_features

def extract_features_from_folder(folder_path):
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4") or f.endswith(".avi")]
    features_list = Parallel(n_jobs=-1)(delayed(extract_features_from_video)(video) for video in video_files)
    return video_files, features_list

# 从第一个文件夹提取特征
video_files1, features_list1 = extract_features_from_folder(input_folder_1)
features_array1 = np.array(features_list1)
print(f"Shape of features_array_1: {features_array1.shape}")

# 从第二个文件夹提取特征
video_files2, features_list2 = extract_features_from_folder(input_folder_2)
features_array2 = np.array(features_list2)
print(f"Shape of features_array_2: {features_array2.shape}")

# 合并两个视频的特征
features_combined = np.concatenate((features_array1, features_array2))
video_files_combined = video_files1 + video_files2

# 进行t-SNE降维
# tsne = TSNE(n_components=2, perplexity=2, n_jobs=4)
# features_2d = tsne.fit_transform(features_combined)

# 进行PCA降维
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_combined)

# 绘制散点图
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(features_2d):
    if i < len(features_array1):
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.01, y + 0.01, os.path.basename(video_files_combined[i]), fontsize=9, color='blue')
    else:
        plt.scatter(x, y, color='red')
        plt.text(x + 0.01, y + 0.01, os.path.basename(video_files_combined[i]), fontsize=9, color='red')
plt.title('Visualizing Latent Space of Videos')
# plt.xlabel('TSNE Component 1')
# plt.ylabel('TSNE Component 2')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(['anger', 'happy'])
plt.show()

# # 进行PCA降维到3D
# pca = PCA(n_components=3)
# features_3d = pca.fit_transform(features_combined)

# # 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for i, (x, y, z) in enumerate(features_3d):
#     if i < len(features_array1):
#         ax.scatter(x, y, z, color='blue')
#         ax.text(x + 0.01, y + 0.01, z + 0.01, os.path.basename(video_files_combined[i]), fontsize=9, color='blue')
#     else:
#         ax.scatter(x, y, z, color='red')
#         ax.text(x + 0.01, y + 0.01, z + 0.01, os.path.basename(video_files_combined[i]), fontsize=9, color='red')

# ax.set_title('Visualizing Latent Space of Videos with 3D PCA')
# ax.set_xlabel('PCA Component 1')
# ax.set_ylabel('PCA Component 2')
# ax.set_zlabel('PCA Component 3')
# ax.legend(['anger', 'happy'])
# plt.show()
# 保存散点图
plt.savefig('test24_2.png')
print("finished")

