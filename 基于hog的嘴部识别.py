import cv2
import dlib
import numpy as np

# 加载人脸检测器和关键点定位器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 加载形态学分割器
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

cap = cv2.VideoCapture(0)


# 定义函数，用于判断嘴巴是否张开
def is_mouth_open(img, mouth_points):
    # 计算嘴巴的形状
    mouth_shape = np.array(mouth_points)
    mouth_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mouth_mask, [mouth_shape], -1, 255, -1)

    # 形态学分割，得到嘴巴图像
    mouth_mask = cv2.morphologyEx(mouth_mask, cv2.MORPH_OPEN, kernel)# 开操作
    mouth_img = cv2.bitwise_and(img, img, mask=mouth_mask)
    # mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('mouth', mouth_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('上嘴唇坐标:', mouth_points[14])
    print('下嘴唇坐标:', mouth_points[18])
    print('上下嘴巴距离:', mouth_points[18][1] - mouth_points[14][1])
    # 模板匹配，得到嘴巴的开合程度
    # res = cv2.matchTemplate(mouth_img, mouth_template, cv2.TM_CCOEFF_NORMED)
    return mouth_points[18][1] - mouth_points[14][1] >= 10, mouth_img

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot capture the frame")
        continue
# 进行人脸检测和关键点定位
    faces = detector(img, 1)
    for face in faces:
        landmarks = predictor(img, face)

        # 获取嘴巴的关键点
        mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

      # 判断嘴巴是否张开
        is_open, mouth_image = is_mouth_open(img, mouth_points)
        # 在图像上绘制嘴巴
        cv2.polylines(img, np.array([mouth_points]), True, (0, 255, 0), 2)
        # print(np.array([mouth_points]))

        # 在图像上显示嘴巴是否张开的结果
        text = 'Open' if is_open else 'Close'
        cv2.putText(img, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# 显示结果
    cv2.imshow('result', img)
    cv2.imshow('mouth', mouth_image)
    # 响应退出键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()