import cv2
#导入图片
img = cv2.imread("sobel.jpg",0)
#计算水平和垂直梯度
gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=1)
gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=1)
#计算梯度图像与角度图像
m,arg=cv2.cartToPolar(gx,gy)
cv2.namedWindow("gx",0)
cv2.resizeWindow("gx", 550, 310)
cv2.namedWindow("gy",0)
cv2.resizeWindow("gy", 550, 310)
cv2.namedWindow("m",0)
cv2.resizeWindow("m", 550, 310)
cv2.namedWindow("arg",0)
cv2.resizeWindow("arg", 550, 310)
cv2.imshow("gx",gx)
cv2.imshow("gy",gy)
cv2.imshow("m",m)
cv2.imshow("arg",arg)
cv2.waitKey(0)