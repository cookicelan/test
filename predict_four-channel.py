from PIL import Image, ImageDraw, ImageFont
import os
import csv
from tqdm import tqdm
from detr import Detection_Transformers
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox


def show_popup(title, message):
    """显示简单的弹窗"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showinfo(title, message)
    root.destroy()


if __name__ == "__main__":

    detr = Detection_Transformers()

    # 只保留目录预测模式
    mode = "dir_predict"

    # -------------------------------------------------------------------------#
    #   阈值参数 - 可在此修改
    # -------------------------------------------------------------------------#
    MIN_WIDTH_THRESHOLD = 100  # bbox宽度阈值
    MIN_HEIGHT_THRESHOLD = 128  # bbox高度阈值
    BOX_NUM = 4

    # -------------------------------------------------------------------------#
    #   路径设置
    # -------------------------------------------------------------------------#
    # dir_origin_path = r'F:\BaiduNetdiskDownload\20251107_白班_png\OK'
    # dir_save_path = r'F:\BaiduNetdiskDownload\20251107_白班_OK_ep268-loss0.458-val_loss0.482_推理结果'
    dir_origin_path = r'F:\BaiduNetdiskDownload\10月29日以来的NG\NG'
    dir_save_path = r'F:\BaiduNetdiskDownload\10月29日以来的NG_ep268-loss0.458-val_loss0.482_推理结果'

    # 结果统计文件路径
    csv_file_path = os.path.join(dir_save_path, os.path.basename(dir_save_path) + ".csv")

    # 中文字体文件路径 - 请替换为实际可用的中文字体文件
    # Windows系统中的黑体字体路径
    chinese_font_path = r"C:\Windows\Fonts\simhei.ttf"
    # 如果找不到黑体，尝试使用宋体
    if not os.path.exists(chinese_font_path):
        chinese_font_path = r"C:\Windows\Fonts\simsun.ttc"

    # 如果还是找不到字体，使用默认字体（可能不支持中文）
    if not os.path.exists(chinese_font_path):
        print("警告: 未找到中文字体文件，中文显示可能不正常")
        chinese_font_path = None

    if mode == "dir_predict":
        # 创建保存目录（如果不存在）
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        # 打开CSV文件准备写入结果 - 添加bbox信息列
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                '文件名',
                'ch1', 'ch1_bbox',
                'ch2', 'ch2_bbox',
                'ch3', 'ch3_bbox',
                'ch4', 'ch4_bbox',
                '总结果'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 初始化统计变量
            total_groups = 0

            # 通道统计：每个通道的OK/轻微油脂/NG数量
            channel_stats = {
                'ch1': {'OK': 0, '轻微油脂': 0, 'NG': 0},
                'ch2': {'OK': 0, '轻微油脂': 0, 'NG': 0},
                'ch3': {'OK': 0, '轻微油脂': 0, 'NG': 0},
                'ch4': {'OK': 0, '轻微油脂': 0, 'NG': 0}
            }

            # 组结果统计
            group_stats = {'OK': 0, '轻微油脂': 0, 'NG': 0}

            # 第一步：收集所有图片并按组分类
            img_groups = defaultdict(list)
            img_names = os.listdir(dir_origin_path)

            for img_name in img_names:
                if img_name.lower().endswith('.png'):
                    # 检查文件名是否包含'ch1', 'ch2', 'ch3', 'ch4'
                    for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
                        if channel in img_name:
                            # 提取组名（去除通道后缀）
                            group_name = img_name.replace(f"_{channel}", "").replace(".png", "")
                            img_groups[group_name].append((channel, img_name))
                            break

            # 第二步：处理每组图片
            for group_name, channel_files in tqdm(img_groups.items(), desc="处理图片组"):
                # 确保每组有4个通道
                if len(channel_files) != 4:
                    print(f"警告: 组 {group_name} 只有 {len(channel_files)} 个通道，跳过处理")
                    continue

                total_groups += 1
                channel_results = {}
                channel_bboxes = {}  # 存储每个通道的bbox信息
                channel_images = []  # 存储每个通道的处理后图像
                group_status = "OK"  # 默认组结果为OK

                # 按通道顺序处理图片
                for channel, img_name in sorted(channel_files):
                    image_path = os.path.join(dir_origin_path, img_name)

                    try:
                        # 打开图片并检查尺寸
                        image = Image.open(image_path)
                        if image.size != (512, 128):
                            raise ValueError(f"图片尺寸不符合要求: {image.size} (应为 512x128)")

                        # 进行目标检测
                        r_image, boxes, labels, confs = detr.detect_image(image, return_results=True)

                        should_save = True
                        # 检查是否满足保存条件：confs小于0.7，至少有1个bbox的宽度（x方向）小于10
                        # should_save = False
                        # if len(boxes) > 0:
                        #     # 检查所有置信度是否都小于0.7
                        #     all_confs_less_than_07 = all(conf < 0.7 for conf in confs)
                        #     # 检查是否有至少一个bbox的宽度小于10
                        #     has_small_width_bbox = any(abs(box[3] - box[1]) < 10 for box in boxes)
                        #     should_save = all_confs_less_than_07 and has_small_width_bbox

                        # 只在满足条件时保存图片和JSON
                        if should_save:
                            # 保存结果图片
                            output_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + '.jpg')
                            r_image.save(output_path, quality=100)

                            # 生成Labelme格式的JSON文件
                            json_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + '.json')
                            try:
                                detr.generate_labelme_json(image_path, boxes, labels, confs, json_path)
                            except Exception as e:
                                print(f"生成JSON文件失败 {json_path}: {e}")
                        else:
                            print(f"跳过保存 {img_name} - 不满足保存条件")

                        # 在图片左上角添加通道描述
                        draw = ImageDraw.Draw(r_image)
                        font_size = 14
                        try:
                            if chinese_font_path and os.path.exists(chinese_font_path):
                                font = ImageFont.truetype(chinese_font_path, font_size)
                            else:
                                # 尝试使用默认字体
                                font = ImageFont.truetype("arial.ttf", font_size)
                        except:
                            # 如果找不到字体，使用默认字体
                            font = ImageFont.load_default()

                        # 设置通道描述文字
                        channel_descriptions = {
                            'ch1': "ch1 径向振动",
                            'ch2': "ch2 轴向振动",
                            'ch3': "ch3 上轴麦克风",
                            'ch4': "ch4 下轴麦克风"
                        }
                        description = channel_descriptions.get(channel, channel)

                        # 计算文字尺寸
                        try:
                            bbox = draw.textbbox((0, 0), description, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except:
                            # 如果textbbox方法不可用，估计文字尺寸
                            text_width = len(description) * font_size
                            text_height = font_size

                        # 添加文字背景（半透明黑色）
                        background = Image.new('RGBA', (text_width + 10, text_height + 5), (0, 0, 0, 128))
                        r_image.paste(background, (5, 5), background)

                        # 绘制文字
                        draw.text((10, 5), description, font=font, fill=(255, 255, 255))

                        # 保存处理后的图像用于组合
                        channel_images.append(r_image.copy())

                        # 判断通道结果状态
                        if len(boxes) == 0:
                            # 没有bbox -> OK
                            channel_status = "OK"
                        else:
                            # 检查所有bbox的尺寸
                            all_slight = True
                            if len(boxes) >= BOX_NUM:
                                channel_status = "NG"
                            else:
                                for box in boxes:
                                    # 计算bbox的宽度和高度
                                    width = abs(box[2] - box[0])
                                    height = abs(box[3] - box[1])

                                    # 如果任何一个bbox超过阈值，则标记为NG
                                    if width >= MIN_WIDTH_THRESHOLD or height >= MIN_HEIGHT_THRESHOLD:
                                        channel_status = "NG"
                                        all_slight = False
                                        break

                                # 如果没有NG，则检查是否所有bbox都是轻微
                                if all_slight:
                                    channel_status = "轻微油脂"

                        # 记录通道结果
                        channel_results[channel] = channel_status

                        # 更新通道统计
                        channel_stats[channel][channel_status] += 1

                        # 收集bbox信息
                        bbox_info = []
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            width = abs(x2 - x1)
                            height = abs(y2 - y1)
                            label = labels[i] if i < len(labels) else "未知"
                            conf = confs[i] if i < len(confs) else 0.0

                            # 格式化bbox信息
                            bbox_str = f"({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) W:{width:.1f} H:{height:.1f} L:{label} C:{conf:.2f}"
                            bbox_info.append(bbox_str)

                        # 将bbox信息存储为字符串
                        channel_bboxes[channel] = "; ".join(bbox_info) if bbox_info else "无"

                    except Exception as e:
                        print(f"处理图片 {img_name} 时出错: {e}")
                        channel_results[channel] = "错误"
                        channel_bboxes[channel] = "错误"
                        # 如果出错，添加一个空白图像占位
                        blank_image = Image.new('RGB', (512, 128), (255, 255, 255))
                        channel_images.append(blank_image)

                # 确定组结果状态
                # 规则：有NG -> NG；无NG但有轻微油脂 -> 轻微油脂；全OK -> OK
                if any(status == "NG" for status in channel_results.values()):
                    group_status = "NG"
                elif any(status == "轻微油脂" for status in channel_results.values()):
                    group_status = "轻微油脂"
                else:
                    group_status = "OK"

                # 更新组统计
                group_stats[group_status] += 1

                # 创建组合图像，计算总高度（4张128像素图片 + 5个2像素间隔）
                spacing = 0  # 间隔像素数
                total_height = 4 * 128 + 5 * spacing
                total_witdh = 512 + 2 * spacing
                combined_image = Image.new('RGB', (total_witdh, total_height), (0, 0, 0))

                # 将4张图从上到下叠放，添加间隔
                x_offset = spacing
                y_offset = spacing

                for i, img in enumerate(channel_images):
                    # 粘贴当前图片（保持原始128像素高度）
                    combined_image.paste(img, (x_offset, y_offset))
                    # 更新y偏移量（图片高度 + 间隔）
                    y_offset += img.height + spacing

                # 在组合图像的右上角添加结果文字
                draw = ImageDraw.Draw(combined_image)

                # 设置字体
                font_size = 30
                try:
                    if chinese_font_path and os.path.exists(chinese_font_path):
                        font = ImageFont.truetype(chinese_font_path, font_size)
                    else:
                        # 尝试使用默认字体
                        font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    # 如果找不到字体，使用默认字体
                    font = ImageFont.load_default()

                # 根据结果设置文字颜色
                if group_status == "OK":
                    text_color = (0, 255, 0)  # 绿色
                elif group_status == "NG":
                    text_color = (255, 0, 0)  # 红色
                else:  # 轻微油脂
                    text_color = (255, 255, 0)  # 黄色

                # 计算文字位置（右上角）
                # 使用draw.textbbox获取文本边界框
                bbox = draw.textbbox((0, 0), group_status, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = total_witdh - text_width - 10  # 距离右边10像素
                text_y = 10  # 距离顶部10像素

                # 添加文字背景（半透明黑色）
                background = Image.new('RGBA', (text_width + 20, text_height + 10), (0, 0, 0, 128))
                combined_image.paste(background, (text_x - 10, text_y - 5), background)

                # 绘制文字
                draw.text((text_x, text_y), group_status, font=font, fill=text_color)

                # 保存组合图像
                combined_path = os.path.join(dir_save_path, f"{group_name}.jpg")
                combined_image.save(combined_path, quality=100)

                # 写入CSV行 - 添加bbox信息
                writer.writerow({
                    '文件名': group_name,
                    'ch1': channel_results.get('ch1', '错误'),
                    'ch1_bbox': channel_bboxes.get('ch1', '错误'),
                    'ch2': channel_results.get('ch2', '错误'),
                    'ch2_bbox': channel_bboxes.get('ch2', '错误'),
                    'ch3': channel_results.get('ch3', '错误'),
                    'ch3_bbox': channel_bboxes.get('ch3', '错误'),
                    'ch4': channel_results.get('ch4', '错误'),
                    'ch4_bbox': channel_bboxes.get('ch4', '错误'),
                    '总结果': group_status
                })

            # 添加总计行
            writer.writerow({
                '文件名': '总计',
                'ch1': f"OK:{channel_stats['ch1']['OK']} 轻微油脂:{channel_stats['ch1']['轻微油脂']} NG:{channel_stats['ch1']['NG']}",
                'ch1_bbox': "",
                'ch2': f"OK:{channel_stats['ch2']['OK']} 轻微油脂:{channel_stats['ch2']['轻微油脂']} NG:{channel_stats['ch2']['NG']}",
                'ch2_bbox': "",
                'ch3': f"OK:{channel_stats['ch3']['OK']} 轻微油脂:{channel_stats['ch3']['轻微油脂']} NG:{channel_stats['ch3']['NG']}",
                'ch3_bbox': "",
                'ch4': f"OK:{channel_stats['ch4']['OK']} 轻微油脂:{channel_stats['ch4']['轻微油脂']} NG:{channel_stats['ch4']['NG']}",
                'ch4_bbox': "",
                '总结果': f"OK:{group_stats['OK']} 轻微油脂:{group_stats['轻微油脂']} NG:{group_stats['NG']}"
            })

            # 添加百分比行
            if total_groups > 0:
                # 计算通道百分比
                ch1_percent = f"OK:{channel_stats['ch1']['OK'] / total_groups:.2%} 轻微油脂:{channel_stats['ch1']['轻微油脂'] / total_groups:.2%} NG:{channel_stats['ch1']['NG'] / total_groups:.2%}"
                ch2_percent = f"OK:{channel_stats['ch2']['OK'] / total_groups:.2%} 轻微油脂:{channel_stats['ch2']['轻微油脂'] / total_groups:.2%} NG:{channel_stats['ch2']['NG'] / total_groups:.2%}"
                ch3_percent = f"OK:{channel_stats['ch3']['OK'] / total_groups:.2%} 轻微油脂:{channel_stats['ch3']['轻微油脂'] / total_groups:.2%} NG:{channel_stats['ch3']['NG'] / total_groups:.2%}"
                ch4_percent = f"OK:{channel_stats['ch4']['OK'] / total_groups:.2%} 轻微油脂:{channel_stats['ch4']['轻微油脂'] / total_groups:.2%} NG:{channel_stats['ch4']['NG'] / total_groups:.2%}"

                # 计算组结果百分比
                group_percent = f"OK:{group_stats['OK'] / total_groups:.2%} 轻微油脂:{group_stats['轻微油脂'] / total_groups:.2%} NG:{group_stats['NG'] / total_groups:.2%}"

                writer.writerow({
                    '文件名': '百分比',
                    'ch1': ch1_percent,
                    'ch1_bbox': "",
                    'ch2': ch2_percent,
                    'ch2_bbox': "",
                    'ch3': ch3_percent,
                    'ch3_bbox': "",
                    'ch4': ch4_percent,
                    'ch4_bbox': "",
                    '总结果': group_percent
                })

        # 输出统计结果
        print("\n检测完成！")
        print(f"原始图片目录: {dir_origin_path}")
        print(f"结果保存目录: {dir_save_path}")
        print(f"结果统计文件: {csv_file_path}")
        print(f"总组数: {total_groups}")

        # 打印通道统计
        for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
            stats = channel_stats[channel]
            print(f"\n{channel} 统计:")
            print(f"  OK: {stats['OK']} ({stats['OK'] / total_groups:.2%})")
            print(f"  轻微油脂: {stats['轻微油脂']} ({stats['轻微油脂'] / total_groups:.2%})")
            print(f"  NG: {stats['NG']} ({stats['NG'] / total_groups:.2%})")

        # 打印组结果统计
        print("\n组结果统计:")
        print(f"  OK: {group_stats['OK']} ({group_stats['OK'] / total_groups:.2%})")
        print(f"  轻微油脂: {group_stats['轻微油脂']} ({group_stats['轻微油脂'] / total_groups:.2%})")
        print(f"  NG: {group_stats['NG']} ({group_stats['NG'] / total_groups:.2%})")

        # 在脚本结束时调用
        # show_popup("处理完成", "文件夹处理操作已成功完成！")

    else:
        # 只保留dir_predict模式，其他模式不再支持
        raise AssertionError("只支持 'dir_predict' 模式")
