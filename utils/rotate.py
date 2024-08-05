import cv2
import dlib
import numpy as np
import os

# 定义处理视频的函数
def process_video(input_path, output_path, predictor_path, history=50, varThreshold=1):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray)
        if len(faces) == 0:
            out.write(frame)
            continue

        face = faces[0]
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 获取眼睛的位置
        left_eye = points[36:42]
        right_eye = points[42:48]

        # 计算眼睛的中心
        left_eye_center = left_eye.mean(axis=0).astype("float")
        right_eye_center = right_eye.mean(axis=0).astype("float")

        # 计算旋转角度
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # 计算旋转矩阵
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                       (left_eye_center[1] + right_eye_center[1]) / 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame_width, frame_height))

        # 使用背景减法分割背景
        fg_mask = bg_subtractor.apply(rotated)

        # 创建一个全黑的掩膜
        mask = np.zeros_like(fg_mask)

        # 使用凸包生成蒙版
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        # 结合背景减法和人脸检测的结果
        combined_mask = cv2.bitwise_and(fg_mask, mask)

        # 使用膨胀和腐蚀操作来改进蒙版
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=2)

        # 保留人脸区域，去掉背景
        result = cv2.bitwise_and(rotated, rotated, mask=combined_mask)

        # 写入处理后的视频帧
        out.write(result)

        # 显示处理后的视频帧（可选）
        cv2.imshow('Result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 设置文件夹路径和模型路径
input_folder = 'data/reanger24'
output_folder = 'outputvideo'
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有视频文件
for filename in os.listdir(input_folder):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f'Processing {input_path} -> {output_path}')
        process_video(input_path, output_path, predictor_path)
