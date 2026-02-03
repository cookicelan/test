import os
import json
import numpy as np
from tqdm import tqdm


def filter_json_files(input_dir, output_dir):
    # -----------------------------------------------------------
    # 1. 初始化
    # -----------------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 已创建输出目录: {output_dir}")

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    total_files = len(json_files)

    if total_files == 0:
        print("❌ 输入文件夹中没有找到 JSON 文件！")
        return

    print(f"📂 原始 JSON 目录: {input_dir}")
    print(f"📂 过滤后保存目录: {output_dir}")
    print(f"🚀 开始处理 {total_files} 个文件...\n")
    print("📋 [最新] 规则执行逻辑:")
    print("   1. [Rule 1] 原始框数量 >= 4 -> 判定为严重NG，全保留 (优先级最高)")
    print("   2. [Rule 2] 数量 < 4 -> 只保留: (H >= 490) 或 (H < 490 且 Center > 384)")
    print("   3. [Rule 3] 过滤后若仅剩 1 个框 且符合 (H < 490 且 Center > 384) -> 剔除\n")

    # 统计数据
    stats = {
        "processed": 0,
        "saved_files": 0,
        "kept_boxes": 0,
        "priority_hit": 0,  # 触发 >=4 规则
        "single_removed": 0,  # 触发单框规则剔除
        "ng_files": 0,
        "ok_files": 0
    }

    # -----------------------------------------------------------
    # 2. 循环处理
    # -----------------------------------------------------------
    for json_file in tqdm(json_files, desc="Processing"):
        stats["processed"] += 1
        input_path = os.path.join(input_dir, json_file)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_shapes = data.get('shapes', [])
            num_original = len(original_shapes)

            final_shapes = []

            # =======================================================
            # 规则 1: 严重NG判定 (优先级最高)
            # 数量 >= 4，全部保留，不进后续逻辑
            # =======================================================
            if num_original >= 4:
                final_shapes = original_shapes
                stats["kept_boxes"] += num_original
                stats["priority_hit"] += 1

            else:
                # ===================================================
                # 规则 2: 几何筛选 (当数量 < 4 时)
                # ===================================================
                candidates = []

                for shape in original_shapes:
                    points = shape.get('points', [])
                    if not points: continue

                    # 几何计算
                    np_points = np.array(points)
                    ys = np_points[:, 1]
                    # 使用 min/max 确保不受画框顺序影响，计算绝对高度
                    height = np.max(ys) - np.min(ys)
                    center_y = np.min(ys) + (height / 2.0)

                    # 判定是否保留
                    should_keep = False

                    # 2.1 高度 >= 490 (长条缺陷，必须保留)
                    if height >= 490:
                        should_keep = True

                    # 2.2 高度 < 490 且 中心点 > 384 (底部短缺陷，暂留)
                    # elif height < 490 and center_y > 384:
                    #     should_keep = True

                    if should_keep:
                        candidates.append(shape)

                # ===================================================
                # 规则 3: 孤立短框过滤 (后处理)
                # 条件: 过滤后只剩 1 个框, 且该框符合 Rule 2.2 (底部短框)
                # ===================================================
                if len(candidates) == 1:
                    # 重新检查这唯一框的属性
                    single_shape = candidates[0]
                    points = single_shape.get('points', [])
                    np_points = np.array(points)
                    ys = np_points[:, 1]
                    h_check = np.max(ys) - np.min(ys)
                    c_check = np.min(ys) + (h_check / 2.0)

                    # 判断它是否属于 "底部短框"
                    # 注意：如果它是长条(>=150)，即便只有一个也要保留，不能剔除
                    is_bottom_short = (h_check < 150 and c_check > 384)

                    if is_bottom_short:
                        final_shapes = []  # 是孤立短框 -> 剔除
                        stats["single_removed"] += 1
                    else:
                        final_shapes = candidates  # 是长条 -> 保留
                        stats["kept_boxes"] += 1

                else:
                    # 如果剩 0 个 或者 >= 2 个框，直接保留筛选结果
                    final_shapes = candidates
                    stats["kept_boxes"] += len(candidates)

            # -----------------------------------------------------------
            # 3. 保存逻辑
            # -----------------------------------------------------------
            data['shapes'] = final_shapes

            if 'imagePath' in data:
                data['imagePath'] = os.path.basename(data['imagePath'])

            output_path = os.path.join(output_dir, json_file)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

            stats["saved_files"] += 1

            if len(final_shapes) > 0:
                stats["ng_files"] += 1
            else:
                stats["ok_files"] += 1

        except Exception as e:
            print(f"⚠️ 文件 {json_file} 处理出错: {e}")

    # -----------------------------------------------------------
    # 4. 输出统计报告
    # -----------------------------------------------------------
    print("\n" + "=" * 50)
    print("📊 过滤统计报告 (Filter Report)")
    print("=" * 50)
    print(f"处理文件总数 : {stats['processed']}")
    print("-" * 50)
    print(f"🔥 [Rule 1] >=4框 全保留 : {stats['priority_hit']} 个文件")
    print(f"🧹 [Rule 3] 单短框被剔除 : {stats['single_removed']} 个文件")
    print("-" * 50)
    print(f"🔴 最终 NG 文件数 : {stats['ng_files']}")
    print(f"🟢 最终 OK 文件数 : {stats['ok_files']}")
    print("=" * 50)
    print(f"✅ 结果已保存在: {output_dir}")


if __name__ == "__main__":
    # ===========================================================
    # 用户自定义接口
    # ===========================================================

    # 1. 原始 JSON 文件夹路径
    INPUT_JSON_DIR = r"E:\李\BRC公司记录\公司安排\NI数据文档\模型推理结果"

    # 2. 过滤后 JSON 保存路径
    OUTPUT_JSON_DIR = r"E:\李\BRC公司记录\公司安排\NI数据文档\添加规则后效果\添加规则后模型效果_V3"

    # ===========================================================

    filter_json_files(INPUT_JSON_DIR, OUTPUT_JSON_DIR)