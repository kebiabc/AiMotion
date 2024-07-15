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
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# from models import models_vit
from model.marlin import Marlin
from joblib import Parallel, delayed


input_folder_1 ='data/test/reanger24'

input_folder_2 = 'data/test/rehappy24'


# 确保模型在GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Marlin.from_file("marlin_vit_base_ytf", "saved/marlin_vit_base_ytf.full.pt").to(device)
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

# 提取文件名末尾的数字
def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.', filename)
    return int(match.group(1)) if match else None

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

labels = [extract_number_from_filename(video_file) for video_file in video_files_combined]

# 进行PCA降维
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_combined)
print(features_2d.shape)
# 线性插值
def linear_interpolate(point1, point2, alpha):
    return point1 * (1 - alpha) + point2 * alpha

# 插值生成新的特征点
new_point = linear_interpolate(features_2d[0], features_2d[10], 0.5)
new_features_high_dim = pca.inverse_transform(new_point)

# 在高维特征空间中进行插值
# high_dim_features_1 = features_array1[0]
# high_dim_features_2 = features_array2[0]
# new_features_high_dim = linear_interpolate(high_dim_features_1, high_dim_features_2, 0.5)

new_features_high_dim_tensor = torch.tensor(new_features_high_dim).unsqueeze(0).to(device)
print(new_features_high_dim_tensor.shape)
# 调用解码器进行视频重构
reconstructed_video_features = model.enc_dec_proj(new_features_high_dim_tensor)
print(reconstructed_video_features.shape)
# reconstructed_video_features = model.decoder.forward_features(reconstructed_video_features)

# # 将解码后的特征转换回视频帧
reconstructed_video = model.decoder.reconstruct_video(reconstructed_video_features)
print(reconstructed_video.shape)
# reconstructed_video = model.decoder.unpatch_to_img(reconstructed_video_features)

# 保存重构的视频
def save_video(frames, output_path, fps):
    frames = frames.permute(0, 2, 3, 4, 1).cpu().detach().numpy()  # 转换为 (B, T, H, W, C) 格式
    frames = (frames * 255).astype(np.uint8)  # 将帧转换为uint8类型
    height, width, layers = frames[0].shape[1:4]
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames[0]:  # 处理每一帧
        out.write(frame)
    out.release()
    print("video saved!")

output_path = 'output_video1.mp4'
fps = 24  # 根据原始视频帧率调整
save_video(reconstructed_video, output_path, fps)

