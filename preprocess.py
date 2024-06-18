
import cv2
import os

# model parameters

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_video(input_path, output_path, size):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于编码视频的编码器
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), size)
    # 循环遍历视频帧直到第10帧
    for i in range(10):
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            exit()
    cap.release()
    # 在第10帧上进行人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        print("人脸坐标：", (x, y, w, h))
    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_region = frame[y:y+h, x:x+w]
    # 调整人脸区域大小为 224x224
        resized_face = cv2.resize(face_region, (224, 224))
        out.write(resized_face)

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

input_folder_1 ='data/test/anger'  # 替换为第一个文件夹输出路径

input_folder_2 = 'data/test/happy'  # 替换为第二个文件夹输出路径

output_folder_1 = 'data/test/reanger'

output_folder_2 = 'data/test/rehappy'
# 处理第一个文件夹
process_videos(input_folder_1, output_folder_1)

# 处理第二个文件夹
process_videos(input_folder_2, output_folder_2)

print("finished")

