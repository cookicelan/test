import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.detr import DETR
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

'''
训练自己的数据集必看注释！
'''


class Detection_Transformers(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'model_data/ep032-loss1.054-val_loss0.825.pth',
        "classes_path": 'model_data/classes_binary.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小
        # ---------------------------------------------------------------------#
        "min_length": 512,
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.7,
        # ---------------------------------------------------------------------#
        #   主干网络的种类
        # ---------------------------------------------------------------------#
        "backbone": 'resnet50',
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化detr
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox()

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        # ---------------------------------------------------#
        #   建立detr模型，载入detr模型的权重
        # ---------------------------------------------------#
        # self.net    = DETR(self.backbone, 'sine', 256, self.num_classes, num_queries=100)
        self.net = DETR(self.backbone, 'sine', 256, self.num_classes, num_queries=100)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False, return_results=False):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, self.min_length)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)

            if results[0] is None:
                return image

            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype='int32')
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(2.5e-2 * image.size[0] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // self.min_length, 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   被包含的bbox过滤，非极大值抑制（NMS）
        # ---------------------------------------------------------#
        # 将(top, left, bottom, right)转换为(x1, y1, x2, y2)格式
        # 注意：top=y1, left=x1, bottom=y2, right=x2
        boxes_for_nms = [(left, top, right, bottom) for top, left, bottom, right in top_boxes]
        # 被包含的bbox过滤
        keep_indices = self.filter_internal_boxes(boxes_for_nms, threshold=0.9)
        # 更新数据
        top_label = top_label[keep_indices]
        top_conf = top_conf[keep_indices]
        top_boxes = top_boxes[keep_indices]

        # boxes_for_nms = [(left, top, right, bottom) for top, left, bottom, right in top_boxes]
        # 应用NMS
        keep_indices = self.nms(boxes_for_nms, top_conf, threshold=0.5)
        # 使用NMS保留的结果更新数据
        top_label = top_label[keep_indices]
        top_conf = top_conf[keep_indices]
        top_boxes = top_boxes[keep_indices]

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # label = '{} {:.2f}'.format(predicted_class, score)
            label = str(int(score * 100))
            draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)
            # 获取文本边界框
            bbox = draw.textbbox((0, 0), label, font=font)
            # 计算文本宽度和高度
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            label_size = (text_width, text_height)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            thickness = 2
            thickness = np.min((thickness, int((right - left) / 3), int((bottom - top) / 3)))
            thickness = np.max((thickness, 1))
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        if return_results:
            # 返回绘制后的图像和检测结果
            return image, top_boxes, top_label, top_conf

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, self.min_length)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()

            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
                if self.cuda:
                    images = images.cuda()
                    images_shape = images_shape.cuda()

                outputs = self.net(images)
                results = self.bbox_util(outputs, images_shape, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, self.min_length)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)

            if results[0] is None:
                return

            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype='int32')
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    # ---------------------------------------------------#
    #   生成PASCAL VOC格式的XML文件
    # ---------------------------------------------------#
    def generate_voc_xml(self, image_path, boxes, labels, confs, output_xml_path):
        """
        生成PASCAL VOC格式的XML文件

        参数:
            image_path: 原始图片路径
            boxes: 检测框列表，格式为[[xmin, ymin, xmax, ymax], ...]
            labels: 类别标签列表
            confs: 置信度列表
            output_xml_path: 输出的XML文件路径
        """
        # 获取图片尺寸
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"❌ 无法读取图片尺寸: {image_path} - {str(e)}")
            return False

        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建XML结构
        annotation = ET.Element("annotation")

        # 文件夹信息
        folder = ET.SubElement(annotation, "folder")
        folder.text = os.path.basename(os.path.dirname(image_path))

        # 文件名
        filename = ET.SubElement(annotation, "filename")
        filename.text = os.path.basename(image_path)

        # 文件路径
        path = ET.SubElement(annotation, "path")
        path.text = os.path.abspath(image_path)

        # 来源信息
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        # 图片尺寸
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"  # 假设RGB图像

        # 分割标记
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        # 创建时间
        created = ET.SubElement(annotation, "created")
        created.text = current_time

        # 添加检测目标
        for i, box in enumerate(boxes):
            obj = ET.SubElement(annotation, "object")

            # 类别名称
            name = ET.SubElement(obj, "name")
            name.text = self.class_names[int(labels[i])]

            # 置信度（VOC格式通常不包含置信度，但我们可以添加为扩展）
            confidence = ET.SubElement(obj, "confidence")
            confidence.text = str(confs[i])

            # 边界框
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(max(0, box[0])))
            ET.SubElement(bndbox, "ymin").text = str(int(max(0, box[1])))
            ET.SubElement(bndbox, "xmax").text = str(int(min(width, box[2])))
            ET.SubElement(bndbox, "ymax").text = str(int(min(height, box[3])))

            # 其他可选信息
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

        # 美化XML格式
        def prettify(elem):
            """返回美化后的XML字符串"""
            rough_string = ET.tostring(elem, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        # 保存XML文件
        try:
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(prettify(annotation))
            print(f"✅ 已生成VOC XML文件: {output_xml_path}")
            return True
        except Exception as e:
            print(f"❌ 保存XML文件错误: {output_xml_path} - {str(e)}")
            return False

    # ---------------------------------------------------#
    #   生成Labelme格式的JSON文件
    # ---------------------------------------------------#
    def generate_labelme_json(self, image_path, boxes, labels, confs, output_json_path):
        """
        生成Labelme格式的JSON文件

        参数:
            image_path: 原始图片路径
            boxes: 检测框列表，格式为[[xmin, ymin, xmax, ymax], ...]
            labels: 类别标签列表
            confs: 置信度列表
            output_json_path: 输出的JSON文件路径（可选）
        """
        # 如果未提供输出路径，则自动生成
        if output_json_path is None:
            # 去掉.jpg后缀再加上.json后缀
            base_path = image_path
            # 处理常见的图片后缀
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                if base_path.lower().endswith(ext):
                    base_path = base_path[:-len(ext)]
                    break
            output_json_path = base_path + '.json'

        # 获取图片尺寸
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"❌ 无法读取图片尺寸: {image_path} - {str(e)}")
            return False

        # 构建Labelme数据结构
        labelme_data = {
            "version": "3.1.1",  # 版本号设为3.1.1
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # 添加检测目标
        for i, box in enumerate(boxes):
            # xmin, ymin, xmax, ymax = box
            ymin, xmin, ymax, xmax = box
            # 创建矩形的四个角点 (左上->右上->右下->左下)
            points = [
                [float(xmin), float(ymin)],  # 左上角
                [float(xmax), float(ymin)],  # 右上角
                [float(xmax), float(ymax)],  # 右下角
                [float(xmin), float(ymax)]  # 左下角
            ]

            shape = {
                "label": self.class_names[int(labels[i])],
                "score": None,  # 设为null
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,  # 注意是布尔值False，不是字符串"false"
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": []
            }
            labelme_data["shapes"].append(shape)

        # 保存JSON文件
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            # print(f"✅ 已生成Labelme JSON文件: {output_json_path}")
            return True
        except Exception as e:
            print(f"❌ 保存JSON文件错误: {output_json_path} - {str(e)}")
            return False

    # ---------------------------------------------------------#
    #   非极大值抑制(NMS)实现
    # ---------------------------------------------------------#
    def nms(self, boxes, scores, threshold=0.5):
        """
        非极大值抑制(Non-Maximum Suppression)

        参数:
            boxes: 边界框列表 [N, 4] (x1, y1, x2, y2)
            scores: 对应的置信度分数 [N]
            threshold: IoU阈值，高于此值的框将被抑制

        返回:
            保留的边界框索引列表
        """
        # 如果没有边界框，直接返回空列表
        if len(boxes) == 0:
            return []

        # 确保boxes是numpy数组
        boxes_array = np.array(boxes, dtype=np.float32)

        # 提取每个框的坐标
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]

        # 计算每个框的面积（加1避免零面积）
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按置信度分数降序排序
        idxs = np.argsort(scores)

        # 初始化保留索引列表
        keep = []

        while len(idxs) > 0:
            # 取出当前最高分的索引
            last = len(idxs) - 1
            i = idxs[last]
            keep.append(i)

            # 获取当前框与其他框的索引
            other_idxs = idxs[:last]

            # 计算当前框与其他所有框的交集坐标
            xx1 = np.maximum(x1[i], x1[other_idxs])
            yy1 = np.maximum(y1[i], y1[other_idxs])
            xx2 = np.minimum(x2[i], x2[other_idxs])
            yy2 = np.minimum(y2[i], y2[other_idxs])

            # 计算交集区域的宽和高（确保非负）
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            # 计算交集面积
            intersection = w * h

            # 计算IoU = 交集面积 / (当前框面积 + 其他框面积 - 交集面积)
            iou = intersection / (areas[i] + areas[other_idxs] - intersection)

            # 找到IoU小于阈值的索引（需要保留）
            suppressed = np.where(iou <= threshold)[0]

            # 更新索引列表，只保留未被抑制的框
            idxs = idxs[suppressed]

        return keep

    def filter_internal_boxes(self, boxes, threshold=0.9):
        """
        过滤几乎被其他边界框完全包含的小边界框

        参数:
            boxes: 边界框列表 [N, 4] (x1, y1, x2, y2)
            threshold: 重叠阈值，如果小框被大框覆盖的比例超过此值，则移除小框

        返回:
            保留的边界框索引列表
        """
        if len(boxes) == 0:
            return []

        n = len(boxes)
        # 确保boxes是numpy数组
        boxes_array = np.array(boxes, dtype=np.float32)

        # 提取每个框的坐标
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 初始化保留标记
        keep = np.ones(n, dtype=bool)  # 初始所有框都保留

        # 按面积降序排序（从大到小）
        sorted_idxs = np.argsort(areas)[::-1]

        for i in range(n):
            if not keep[sorted_idxs[i]]:
                continue  # 如果已经被标记为移除，跳过

            for j in range(i + 1, n):
                if not keep[sorted_idxs[j]]:
                    continue  # 如果已经被标记为移除，跳过

                # 获取两个框的索引
                idx_i = sorted_idxs[i]
                idx_j = sorted_idxs[j]

                # 计算交集
                inter_x1 = max(x1[idx_i], x1[idx_j])
                inter_y1 = max(y1[idx_i], y1[idx_j])
                inter_x2 = min(x2[idx_i], x2[idx_j])
                inter_y2 = min(y2[idx_i], y2[idx_j])

                inter_w = max(0, inter_x2 - inter_x1 + 1)
                inter_h = max(0, inter_y2 - inter_y1 + 1)
                inter_area = inter_w * inter_h

                # 计算框i被框j覆盖的比例
                coverage_i = inter_area / areas[idx_i]
                # 计算框j被框i覆盖的比例
                coverage_j = inter_area / areas[idx_j]

                # 如果框j几乎完全包含在框i中
                if coverage_j > threshold:
                    keep[idx_j] = False  # 移除小框j

                # 如果框i几乎完全包含在框j中
                elif coverage_i > threshold:
                    keep[idx_i] = False  # 移除小框i
                    break  # 跳出内层循环，继续处理下一个框

        # 返回保留框的索引
        return np.where(keep)[0].tolist()
