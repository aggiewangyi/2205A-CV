import cv2

# 加载预训练的 Haar Cascade 分类器
vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# 读取图像
image_path = 'your_image.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

# 检测车辆
vehicles = vehicle_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 绘制检测框
for (x, y, w, h) in vehicles:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Vehicle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()