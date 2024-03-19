import cv2
import numpy as np

# 定义各种参数和常量
MOUTH_START = 48
MOUTH_END = 68
LANDMARKS = list(range(MOUTH_START, MOUTH_END))
# 嘴巴张开关闭阈值
CLOSED_MOUTH_THRESH = 25
OPEN_MOUTH_THRESH = 50

# 加载分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot capture the frame")
        continue

    # 转换成灰色图像以减少计算复杂性
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 在图像中检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # 如果找到一个或多个人脸，则为每个人脸绘制矩形并查找嘴巴
    for (x, y, w, h) in faces:
        # 绘制人脸矩形
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 在人脸区域中搜索嘴巴
        mouth_rect = [0, 0, 0, 0]
        mouth_found = False
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # 搜索嘴唇矩形
        mouths = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml').detectMultiScale(roi_gray, scaleFactor=1.3,
                                                                                     minNeighbors=5)
        for (mx, my, mw, mh) in mouths:
            if (16 * my > mh) and (112 * my < 65 * mh):
                mouth_rect = [mx, my, mw, mh]
                mouth_found = True
                break

        # 如果找到嘴巴，则确定嘴部的张开程度并输出信息
        if mouth_found:
            # 计算嘴部高度与宽度比率并标准化
            mouth_ratio = float(mouth_rect[3]) / float(w)
            normalized_ratio = (mouth_ratio - CLOSED_MOUTH_THRESH) / (OPEN_MOUTH_THRESH - CLOSED_MOUTH_THRESH)

            if mouth_rect[2] * mouth_rect[3] < 800:  # 嘴巴区域面积小于800则视为未检测到嘴巴
                state_text = 'Unknown'
                state_color = (128, 128, 128)  # 灰色
            elif normalized_ratio <= 0:
                state_text = 'Mouth Closed'
                state_color = (0, 0, 255)  # 红色
            elif normalized_ratio >= 1:
                state_text = 'Mouth Open'
                state_color = (0, 255, 0)  # 绿色
            else:
                state_text = 'Mouth Partially Open'
                state_color = (0, 255, 255)  # 黄色

            # 显示口部状态文本
            cv2.putText(img, state_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, thickness=2)
        else:
            # 如果没有检测到嘴巴，则将预测状态设置为“未知”，并使用灰色文本显示。
            state_text = 'Unknown'
            state_color = (128, 128, 128)  # 灰色
            cv2.putText(img, state_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, thickness=2)
    # 显示图像
    cv2.imshow('Img', img)
    # 响应退出键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()