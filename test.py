import cv2

# 加载模板图片
template = cv2.imread("close_template.jpg", 0) #读取灰度图像
if template is None:
    print("Error: Failed to read template image!") #输出错误消息并退出程序

# 加载测试图片
test_image = cv2.imread("test_open.jpg")
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# 使用 cv2.matchTemplate 进行模板匹配操作
res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #得到最大匹配值与对应位置

# 根据阈值判断匹配结果
threshold = 0.8
if max_val > threshold:
    print("匹配成功！")
    # 进一步处理匹配结果...
else:
    print("未找到对应目标。")