import mysql.connector

# # 连接 MySQL 数据库
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="root",
#     database="image_search_db"
# )
# mycursor = mydb.cursor()
#
#
# # 假设你有一个记录关联关系的 JSON 文件
# with open(r'C:\Users\26296\Desktop\Chinese-CLIP-master\DATAPATH\datasets\test_predictions.jsonl', 'r') as f:
#     relation_data = []
#     for line in f:
#         relation_data.append(json.loads(line))
#     # 遍历关联数据并插入到数据库
#     for item in relation_data:
#         image_ids = item["image_ids"]
#         text_id = item["text_id"]
#         for image_id in image_ids:
#             insert_query = "INSERT INTO image_text_relation (image_id, text_id) VALUES (%s, %s)"
#             values = (image_id, text_id)
#             mycursor.execute(insert_query, values)
#
# mydb.commit()
# mycursor.close()
# mydb.close()

