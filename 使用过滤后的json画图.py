import os
import json
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def visualize_and_stat(img_dir, json_dir, save_dir):
    # -----------------------------------------------------------
    # 1. 初始化与准备
    # -----------------------------------------------------------
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✅ 已创建保存目录: {save_dir}")

    # 支持的图片格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # 获取原始文件夹下所有的图片文件 (作为统计的分母)
    all_img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
    total_images_count = len(all_img_files)

    if total_images_count == 0:
        print("❌ 错误：原始图片文件夹为空！")
        return

    # --- 初始化统计数据 ---
    stats = {
        "gt_ok_count": 0,  # 实际 OK 总数 (分母)
        "gt_ng_count": 0,  # 实际 NG 总数 (分母)
        "pred_ng_count": 0,  # 预测为 NG 的图片数 (用于画图统计)
        "total_ng_boxes": 0,  # 绘制的框总数

        "overkill_files": [],  # 过杀文件列表 (OK 被判 NG)
        "leakage_files": []  # 漏检文件列表 (NG 被判 OK)
    }

    print(f"📂 原始图片: {img_dir}")
    print(f"📂 JSON数据: {json_dir}")
    print(f"🚀 开始处理 {total_images_count} 张图片...\n")

    # -----------------------------------------------------------
    # 2. 循环处理每一张图片
    # -----------------------------------------------------------
    for img_file in tqdm(all_img_files, desc="Processing"):
        image_path = os.path.join(img_dir, img_file)
        file_name_no_ext = os.path.splitext(img_file)[0]
        json_path = os.path.join(json_dir, file_name_no_ext + ".json")

        # =======================================================
        # A. 确定 真值 (Ground Truth)
        # =======================================================
        # 规则：文件名包含 "OK" (忽略大小写) 则为 OK，否则为 NG
        is_gt_ok = "OK" in img_file.upper()

        if is_gt_ok:
            stats["gt_ok_count"] += 1
        else:
            stats["gt_ng_count"] += 1

        # =======================================================
        # B. 确定 预测值 (Prediction)
        # =======================================================
        is_pred_ng = False
        shapes = []

        # 检查 JSON 是否存在且包含有效的 shapes
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                shapes = data.get('shapes', [])
                if shapes:
                    is_pred_ng = True
            except Exception:
                pass  # JSON 读取失败或格式错误，视为 OK (无框)

        # =======================================================
        # C. 统计逻辑 (核心逻辑)
        # =======================================================

        # 1. 过杀 (Overkill): 真值是 OK，但模型预测是 NG
        if is_gt_ok and is_pred_ng:
            stats["overkill_files"].append(img_file)

        # 2. 漏检 (Leakage): 真值是 NG，但模型预测是 OK
        elif not is_gt_ok and not is_pred_ng:
            stats["leakage_files"].append(img_file)

        # =======================================================
        # D. 绘制逻辑 (仅对预测为 NG 的图片进行绘制和保存)
        # =======================================================
        if is_pred_ng:
            stats["pred_ng_count"] += 1
            try:
                # 打开图片
                image = Image.open(image_path).convert('RGB')
                draw = ImageDraw.Draw(image)

                # --- 样式设置 ---
                font_size = max(15, int(image.size[1] * 0.03))
                try:
                    font = ImageFont.truetype("simhei.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()
                line_width = max(2, int(image.size[0] * 0.005))

                # 绘制所有框
                for shape in shapes:
                    label = shape.get('label', 'NG')
                    points = shape.get('points', [])
                    if not points: continue

                    np_points = np.array(points)
                    x1 = np.min(np_points[:, 0])
                    y1 = np.min(np_points[:, 1])
                    x2 = np.max(np_points[:, 0])
                    y2 = np.max(np_points[:, 1])

                    # 画框
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=line_width)

                    # 画标签
                    text_content = f"{label}"
                    if hasattr(draw, 'textbbox'):
                        bbox = draw.textbbox((0, 0), text_content, font=font)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(text_content, font=font)

                    text_x = x1
                    text_y = y1 - text_h if y1 - text_h >= 0 else y1
                    draw.rectangle([text_x, text_y, text_x + text_w, text_y + text_h], fill='red')
                    draw.text((text_x, text_y), text_content, fill='white', font=font)

                    stats["total_ng_boxes"] += 1

                # 保存绘制好的图片 (过杀的图也会被画出来，方便分析)
                save_path = os.path.join(save_dir, img_file)
                image.save(save_path, quality=95)

            except Exception as e:
                print(f"⚠️ 绘图出错 {img_file}: {e}")

    # -----------------------------------------------------------
    # 3. 计算比率
    # -----------------------------------------------------------
    overkill_count = len(stats["overkill_files"])
    leakage_count = len(stats["leakage_files"])

    # 过杀率 = 过杀数量 / 实际OK总数
    overkill_rate = 0.0
    if stats["gt_ok_count"] > 0:
        overkill_rate = (overkill_count / stats["gt_ok_count"]) * 100

    # 漏检率 = 漏检数量 / 实际NG总数
    leakage_rate = 0.0
    if stats["gt_ng_count"] > 0:
        leakage_rate = (leakage_count / stats["gt_ng_count"]) * 100

    # -----------------------------------------------------------
    # 4. 生成 CSV 统计报告
    # -----------------------------------------------------------
    csv_filename = "统计报告.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    try:
        # 使用 utf-8-sig 编码，防止 Excel 打开中文乱码
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)

            # 写入汇总信息
            writer.writerow(["=== 汇总统计 (Summary) ==="])
            writer.writerow(["指标 (Metric)", "数值 (Value)", "说明 (Note)"])
            writer.writerow(["总图片数", total_images_count, ""])
            writer.writerow(["实际 OK 总数", stats["gt_ok_count"], "Filename contains 'OK'"])
            writer.writerow(["实际 NG 总数", stats["gt_ng_count"], "Filename without 'OK'"])
            writer.writerow([])
            writer.writerow(["预测 NG 图片数", stats["pred_ng_count"], "Model detected NG"])
            writer.writerow(["预测 NG 框总数", stats["total_ng_boxes"], "Total boxes drawn"])
            writer.writerow([])
            writer.writerow(["过杀数量 (Overkill)", overkill_count, "True OK -> Pred NG"])
            writer.writerow(["漏检数量 (Leakage)", leakage_count, "True NG -> Pred OK"])
            writer.writerow(["过杀率 (Overkill Rate)", f"{overkill_rate:.2f}%", "Overkill / Actual OK"])
            writer.writerow(["漏检率 (Leakage Rate)", f"{leakage_rate:.2f}%", "Leakage / Actual NG"])
            writer.writerow([])
            writer.writerow([])

            # 写入详细过杀名单
            writer.writerow(["=== 过杀详细名单 (Overkill List) ==="])
            writer.writerow(["文件名 (Filename)", "错误类型 (Type)"])
            if overkill_count == 0:
                writer.writerow(["无过杀", "-"])
            else:
                for name in stats["overkill_files"]:
                    writer.writerow([name, "过杀 (Overkill)"])

            writer.writerow([])

            # 写入详细漏检名单
            writer.writerow(["=== 漏检详细名单 (Leakage List) ==="])
            writer.writerow(["文件名 (Filename)", "错误类型 (Type)"])
            if leakage_count == 0:
                writer.writerow(["无漏检", "-"])
            else:
                for name in stats["leakage_files"]:
                    writer.writerow([name, "漏检 (Leakage)"])

        print(f"📄 CSV 统计表格已生成: {csv_path}")

    except Exception as e:
        print(f"❌ 生成CSV失败: {e}")

    # -----------------------------------------------------------
    # 5. 控制台输出简报
    # -----------------------------------------------------------
    print("\n" + "=" * 50)
    print("📊 统计结果简报 (Report)")
    print("=" * 50)
    print(f"实际 OK : {stats['gt_ok_count']}")
    print(f"实际 NG : {stats['gt_ng_count']}")
    print("-" * 50)
    print(f"🚫 过杀 (Overkill) : {overkill_count} 张")
    print(f"   过杀率 : {overkill_rate:.2f}%")
    print(f"⚠️ 漏检 (Leakage)  : {leakage_count} 张")
    print(f"   漏检率 : {leakage_rate:.2f}%")
    print("=" * 50)
    print(f"详细结果请查看目录下的: {csv_filename}")


if __name__ == "__main__":
    # ===========================================================
    # 用户自定义配置区域 (请在这里修改路径)
    # ===========================================================

    # 1. 原始图片存放地址
    ORIGIN_IMG_PATH = r"E:\李\BRC公司记录\公司安排\NI数据文档\NI归一化时频图"

    # 2. JSON 文件存放地址 (使用规则过滤后的JSON文件夹)
    JSON_PATH = r"E:\李\BRC公司记录\公司安排\NI数据文档\添加规则后效果\添加规则后模型效果_V3"

    # 3. 结果保存地址 (图片 + CSV)
    OUTPUT_SAVE_PATH = r"E:\李\BRC公司记录\公司安排\NI数据文档\添加规则后效果\添加规则后模型效果_V3"

    # ===========================================================

    visualize_and_stat(ORIGIN_IMG_PATH, JSON_PATH, OUTPUT_SAVE_PATH)