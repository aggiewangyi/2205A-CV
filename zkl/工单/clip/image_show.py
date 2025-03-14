import lmdb
from PIL import Image
import base64
from io import BytesIO
import os

def show_image(image_ids):
    lmdb_open = lmdb.open(r"C:\Users\26296\Desktop\Chinese-CLIP-master\DATAPATH\datasets\MUGE\lmdb\test\imgs", readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_imgs = lmdb_open.begin(buffers=True)
    cursor_imgs = txn_imgs.cursor()

    #判断文件夹是否为空，如果不为空则清空
    items = os.listdir('./result')
    if items:
        # 如果文件夹不为空，遍历其中的所有文件和子文件夹
        for item in items:
            item_path = os.path.join('./result', item)
            if os.path.isfile(item_path):
                # 如果是文件，直接删除
                os.remove(item_path)
        print(f"文件夹已清空。")
    else:
        print(f"文件夹本来就是空的。")

    image_paths = []
    # 查找的图片 ID
    for i,target_img_id in enumerate(image_ids):
        target_img_id = target_img_id.encode()

        # 使用游标定位到指定的图片 ID
        if cursor_imgs.set_key(target_img_id):
            # 获取对应的值（图片的 Base64 编码数据）
            image_b64 = cursor_imgs.value()
            image_b64 = image_b64.tobytes()
            try:
                image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
                # 解码 Base64 数据并打开图像
                image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
                # 保存图片
                image.save("./result/"+str(i)+".png")
                image_paths.append("./result/"+str(i)+".png")

            except Exception as e:
                print(f"解码图片时出现错误: {e}")
        else:
            print(f"未找到 ID 为 {target_img_id} 的图片。")

    # 关闭事务和数据库
    txn_imgs.abort()
    lmdb_open.close()
    return image_paths