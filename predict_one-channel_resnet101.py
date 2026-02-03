from PIL import Image
import os
from tqdm import tqdm
from detr_resnet101 import Detection_Transformers

if __name__ == "__main__":
    detr = Detection_Transformers()

    # 只保留目录预测模式
    mode = "dir_predict"

    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   result_txt_path     指定结果TXT文件的保存路径
    # -------------------------------------------------------------------------#
    dir_origin_path = r"E:\MFCC_ResNet模型\detr-pytorch_one-channel_binary\VOCdevkit\VOC2007\JPEGImages"
    dir_save_path = r"E:\MFCC_ResNet模型\detr-pytorch_one-channel_binary\VOCdevkit\VOC2007\JPEGImages_推理结果_ep285-loss0.810-val_loss0.935_置信阈值=0.7"
    result_txt_path = os.path.join(dir_save_path, os.path.basename(dir_save_path) + ".txt")

    # 确保保存目录存在
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
        print(f"已创建保存目录: {dir_save_path}")

    if mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        count = 0
        total = 0

        # 创建结果TXT文件
        with open(result_txt_path, 'w', encoding='gbk') as result_file:
            for img_name in tqdm(img_names):
                # 支持多种图片格式
                if img_name.lower().endswith('.png'):
                    image_path = os.path.join(dir_origin_path, img_name)
                    try:
                        image = Image.open(image_path)
                    except Exception as e:
                        print(f"无法打开图片 {image_path}: {e}")
                        continue

                    # 进行目标检测并获取结果
                    r_image, boxes, labels, confs = detr.detect_image(image, return_results=True)

                    # 创建保存目录（如果不存在）
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)

                    # 生成Labelme格式的JSON文件
                    json_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + '.json')
                    try:
                        detr.generate_labelme_json(image_path, boxes, labels, confs, json_path)
                    except Exception as e:
                        print(f"生成JSON文件失败 {json_path}: {e}")

                    # 保存结果图片
                    try:
                        # 保持原始图片格式
                        output_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + '.jpg')
                        r_image.save(output_path, quality=100)
                    except Exception as e:
                        print(f"保存结果图片失败 {output_path}: {e}")

                    # 统计检测结果
                    if len(boxes) > 0:
                        count += 1
                    total += 1

                    # 将结果写入TXT文件
                    # 写入图片路径
                    result_file.write(image_path)

                    # 写入检测到的bbox
                    for i, box in enumerate(boxes):
                        # 确保box是整数坐标
                        x1, y1, x2, y2 = map(int, box[:4])
                        class_id = int(labels[i]) if i < len(labels) else 0
                        result_file.write(f" {x1},{y1},{x2},{y2},{class_id}")

                    # 换行
                    result_file.write("\n")

        # 输出统计结果
        print("\n检测完成！")
        print(f"原始图片目录: {dir_origin_path}")
        print(f"结果保存目录: {dir_save_path}")
        print(f"结果TXT文件: {result_txt_path}")
        print(f"总图片数量: {total}")
        print(f"检测到目标的图片数量: {count}")
        print(f"检测到目标的图片比例: {count / total:.2%}")

    else:
        # 只保留dir_predict模式，其他模式不再支持
        raise AssertionError("只支持 'dir_predict' 模式")
