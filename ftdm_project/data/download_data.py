#!/usr/bin/env python3
# data/download_data.py
# 自动下载论文使用的 UCI 意大利空气质量数据集
#
# 论文注脚1：https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci
# UCI 原始来源：https://archive.ics.uci.edu/dataset/360/air+quality
#
# 本脚本尝试从多个镜像源自动下载，
# 失败时会给出手动下载的详细步骤。

import os
import sys
import urllib.request
import zipfile
import shutil

# 保存路径（相对于 ftdm_project/）
SAVE_DIR     = os.path.join(os.path.dirname(__file__))
TARGET_FILE  = os.path.join(SAVE_DIR, 'AirQualityUCI.csv')
TARGET_XLSX  = os.path.join(SAVE_DIR, 'AirQualityUCI.xlsx')

# UCI 数据集直链（ZIP 包含 .xlsx 原始文件）
UCI_ZIP_URL = (
    'https://archive.ics.uci.edu/static/public/360/air+quality.zip'
)


def show_manual_instructions():
    print()
    print("=" * 60)
    print("自动下载失败，请手动下载数据集：")
    print("=" * 60)
    print()
    print("方法 1（推荐）— Kaggle：")
    print("  网址：https://www.kaggle.com/datasets/aayushkandpal/"
          "air-quality-time-series-data-uci")
    print("  下载后将 AirQualityUCI.csv 放到：")
    print(f"    {SAVE_DIR}/")
    print()
    print("方法 2 — UCI 官网：")
    print("  网址：https://archive.ics.uci.edu/dataset/360/air+quality")
    print("  下载 ZIP，解压后将 AirQualityUCI.xlsx 转为 CSV 格式，")
    print("  文件名改为 AirQualityUCI.csv，放到：")
    print(f"    {SAVE_DIR}/")
    print()
    print("数据集说明：")
    print("  - 9357 条小时均值感知记录")
    print("  - 意大利某城市空气质量化学传感器数据")
    print("  - 包含 CO, PT08.S1, C6H6, PT08.S2, NOx, NO2, T 等列")
    print("  - 异常值标记为 -200")
    print()


def download_from_uci():
    """尝试从 UCI 官网下载并转换"""
    print(f"正在从 UCI 下载: {UCI_ZIP_URL}")

    zip_path = os.path.join(SAVE_DIR, '_tmp_air_quality.zip')

    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        print("下载完成，正在解压...")

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SAVE_DIR)

        os.remove(zip_path)

        # UCI zip 中包含 AirQualityUCI.xlsx 和 AirQualityUCI.csv
        csv_candidates = [
            os.path.join(SAVE_DIR, 'AirQualityUCI.csv'),
            os.path.join(SAVE_DIR, 'AirQualityUCI.xlsx'),
        ]

        for f in csv_candidates:
            if os.path.exists(f):
                print(f"找到文件: {f}")
                if f.endswith('.xlsx'):
                    convert_xlsx_to_csv(f)
                return True

        print("ZIP 中未找到期望文件")
        return False

    except Exception as e:
        print(f"下载失败: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


def convert_xlsx_to_csv(xlsx_path: str):
    """将 xlsx 文件转换为 CSV（用于 UCI 原始格式）"""
    try:
        import pandas as pd
        print(f"正在将 {xlsx_path} 转换为 CSV...")
        df = pd.read_excel(xlsx_path, sheet_name=0)
        csv_path = xlsx_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV 已保存至: {csv_path}")
    except ImportError:
        print("需要安装 openpyxl: pip install openpyxl")
        raise


def check_existing():
    """检查是否已有数据文件"""
    for f in [TARGET_FILE, TARGET_XLSX]:
        if os.path.exists(f):
            print(f"数据文件已存在: {f}")
            return True
    return False


def validate_csv(csv_path: str) -> bool:
    """简单验证 CSV 文件格式是否符合预期"""
    try:
        import pandas as pd
        # 尝试多种分隔符
        for sep, decimal in [(';', ','), (',', '.'), ('\t', '.')]:
            try:
                df = pd.read_csv(csv_path, sep=sep, decimal=decimal, nrows=5)
                if df.shape[1] >= 10:
                    # 检查关键列
                    cols = [c.strip() for c in df.columns]
                    required = ['CO(GT)', 'NOx(GT)', 'T']
                    found = [r for r in required if r in cols]
                    if len(found) >= 2:
                        print(f"  数据验证通过，检测到列: {cols[:6]}...")
                        return True
            except Exception:
                continue
        print("警告：数据文件格式可能不正确，请检查列名")
        return False
    except ImportError:
        print("提示：安装 pandas 后可自动验证文件格式")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("UCI 空气质量数据集下载工具")
    print("论文：A Coverage-Aware High-Quality Sensing Data")
    print("      Collection Method in Mobile Crowd Sensing")
    print("=" * 60)

    if check_existing():
        validate_csv(TARGET_FILE)
        print("\n无需重新下载。")
        sys.exit(0)

    print("\n尝试自动下载...")
    success = download_from_uci()

    if success and os.path.exists(TARGET_FILE):
        print("\n下载成功！")
        validate_csv(TARGET_FILE)
        print(f"\n接下来运行：")
        print(f"  cd ..")
        print(f"  python main.py --data_path data/AirQualityUCI.csv --quick")
    else:
        show_manual_instructions()
        sys.exit(1)
