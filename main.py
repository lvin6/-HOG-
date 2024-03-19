import cv2
import numpy as np

# 定义不张嘴和张嘴模板
mouth_closed_template = cv2.imread('close_template.jpg', 0)
mouth_opened_template = cv2.imread('open_template.jpg', 0)

cap = cv2.VideoCapture(0)

while(True):
    # 读取视频流并转换成灰度图像
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行嘴部的模板匹配
    mouth_closed_result = cv2.matchTemplate(gray, mouth_closed_template, cv2.TM_CCOEFF_NORMED)
    mouth_opened_result = cv2.matchTemplate(gray, mouth_opened_template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.5

    # 检测嘴被张开或关闭
    if (mouth_closed_result.max() > threshold) and (mouth_opened_result.max() < threshold):
        state_text = 'Mouth Closed'
        state_color = (0, 0, 255) # 红色
    elif (mouth_opened_result.max() > threshold) and (mouth_closed_result.max() < threshold):
        state_text = 'Mouth Open'
        state_color = (0, 255, 0) # 绿色
    else:
        state_text = 'unknown'
        state_color = (0, 0, 0) # 黑色

    # 绘制状态文本
    cv2.putText(frame, state_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, thickness=2)

    # 将处理后的图像显示到屏幕上
    cv2.imshow('Facial Mouth Detector', frame)

    # 当用户按下'q'键时退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭视频流，并销毁所有窗口
cap.release()
cv2.destroyAllWindows()