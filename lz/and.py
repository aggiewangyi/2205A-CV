import cv2
import dlib
import numpy as np


def detect_face(image_path):
    # 加载检测器
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces, img


def expand_and_rotate(image_path, output_path, expand_pixels=50, rotate_angle=0):
    # 1. 检测面部
    faces, img = detect_face(image_path)
    if len(faces) == 0:
        print("未检测到面部！")
        return

    # 2. 扩展面部区域
    face = faces[0]  # 取第一个检测到的面部
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    expanded_face = img[
                    max(y - expand_pixels, 0): min(y + h + expand_pixels, img.shape[0]),
                    max(x - expand_pixels, 0): min(x + w + expand_pixels, img.shape[1])
                    ]

    # 3. 修复扩展区域（示例：简单模糊处理，可替换为AI修复模型）
    mask = np.zeros(expanded_face.shape[:2], dtype=np.uint8)
    mask[expand_pixels:-expand_pixels, expand_pixels:-expand_pixels] = 255
    expanded_face = cv2.inpaint(expanded_face, mask, 3, cv2.INPAINT_TELEA)

    # 4. 旋转图像
    (h, w) = expanded_face.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(expanded_face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 保存结果
    cv2.imwrite(output_path, rotated)
    print(f"结果已保存至 {output_path}")


# 使用示例
if __name__ == "__main__":
    expand_and_rotate("input.jpg", "output_rotated.jpg", expand_pixels=50, rotate_angle=30)