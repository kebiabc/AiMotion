import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# 使用Multicore-TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.decomposition import PCA
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

def center_crop(frame, size):
    height, width, _ = frame.shape
    new_width, new_height = size

    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)
    right = left + new_width
    bottom = top + new_height

    cropped_frame = frame[top:bottom, left:right]
    return cropped_frame

def crop_video(input_path, output_path, size):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于编码视频的编码器
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = center_crop(frame, size)
        out.write(cropped_frame)

    cap.release()
    out.release()

def process_videos(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):  # 根据需要可以添加更多的视频格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, "resized_" + filename)
            crop_video(input_path, output_path, size)

# input_folder_1 ='data/winetype1'  # 替换为第一个文件夹输出路径

# input_folder_2 = 'data/water'  # 替换为第二个文件夹输出路径

# output_folder_1 = 'data/resizedwinetype1'

# output_folder_2 = 'data/resizedwater1'
# # 处理第一个文件夹
# process_videos(input_folder_1, output_folder_1)

# # 处理第二个文件夹
# process_videos(input_folder_2, output_folder_2)

input_folder_1 ='data/resizedwine' 

input_folder_2 = 'data/resizedwater' 

# # create model
# model = Marlin.from_file("marlin_vit_base_ytf", "saved/marlin_vit_base_ytf.encoder.pt").to(device)
# print(model)

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

def extract_label(video_file):
    # 使用正则表达式查找 subject 和 _product 之间的部分
    match = re.search(r'subject(.*?)_product', os.path.basename(video_file))
    return match.group(1) if match else 'Unknown'


labels = [extract_label(video_file) for video_file in video_files_combined]

# 进行t-SNE降维
tsne = TSNE(n_components=2, perplexity=10, n_jobs=-1)
features_2d = tsne.fit_transform(features_combined)
# 计算每个标签对应的蓝点中点
blue_points = features_2d[:len(features_array1)]
blue_labels = labels[:len(features_array1)]
red_points = features_2d[len(features_array1):]
red_labels = labels[len(features_array1):]

label_to_mean = {}
for label in set(blue_labels):
    points = blue_points[np.array(blue_labels) == label]
    mean_point = np.mean(points, axis=0)
    label_to_mean[label] = mean_point

# 绘制散点图和箭头
angles = []

plt.figure(figsize=(12, 8))
blue_scatter = plt.scatter([], [], color='blue', label='wine')
red_scatter = plt.scatter([], [], color='red', label='water')
yellow_scatter = plt.scatter([], [], color='yellow', label='Mean Wine')

for i, (x, y) in enumerate(blue_points):
    plt.scatter(x, y, color='blue')

for i, (x, y) in enumerate(red_points):
    mean_point = label_to_mean.get(red_labels[i])
    if mean_point is not None:
        # 绘制从红点到黄点的箭头
        dx = mean_point[0] - x
        dy = mean_point[1] - y
        plt.arrow(x, y, dx, dy, color='black', head_width=0.05, head_length=0.1)
        # 计算角度
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    plt.scatter(x, y, color='red')

# 绘制每个标签对应的蓝点中点
for mean_point in label_to_mean.values():
    plt.scatter(mean_point[0], mean_point[1], color='yellow', s=100)

plt.title('Visualizing Latent Space of Videos')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.legend(handles=[blue_scatter, red_scatter, yellow_scatter])

# # 计算平均角度
# mean_angle = np.mean(angles)
# mean_angle_degrees = np.degrees(mean_angle)
# print(f"Average angle: {mean_angle_degrees:.2f} degrees")
var_angle = np.var(angles)
var_angle_degrees = np.degrees(var_angle)
print(f"Variance angle: {var_angle_degrees:.2f} degrees")
plt.show()

# 保存散点图
plt.savefig('wineandwater5.png')
print("finished")

