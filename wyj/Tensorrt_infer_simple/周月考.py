
#像在获取过程中，由于各种因素如传感器的热噪声、环境中的电磁干扰等，会产生噪声，
#因此图像预处理在图像处理和计算机视觉等众多领域中都具有极其重要的价值


#1、导入图像预处理需要用到的相关库;
#2、定义一个图像预处理函数image_process;
#3、给图像预处理函数传入需要处理的图像路径；
#4、在图像预处理函数中根据给定的图像路径读取图像；
#5、使用图像处理包将图像缩放到合适的大小；
#6、使用图像处理包显示原始图像；
#7、按照显示原图像的方式显示缩放后的图像；
#8、将缩放后的图像保存起来；
#9、在代码合适的地方添加注释；


#1、导入图像预处理需要用到的相关库;
import cv2

#2、定义一个图像预处理函数image_process;
#3、给图像预处理函数传入需要处理的图像路径；
def image_process(image_path):
    # 4、在图像预处理函数中根据给定的图像路径读取图像；
    src = cv2.imread(image_path)
    # 5、使用图像处理包将图像缩放到合适的大小；
    dst = cv2.resize(src, (640, 640))
    # 6、使用图像处理包显示原始图像；
    cv2.imshow("org", src)
    # 7、按照显示原图像的方式显示缩放后的图像；
    cv2.imshow("resize", dst)
    cv2.waitKey(0)
    # 8、将缩放后的图像保存起来；
    cv2.imwrite("resized_image.jpg", dst)


if __name__ == "__main__":
    image_path = "images\street.jpg"
    image_process(image_path)

