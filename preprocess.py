import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def crop_image_from_gray(img, tol=7):
    """裁剪眼底图像的无用黑边"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img):
    """圆形 ROI 裁剪，去除边缘光晕和暗角干扰"""
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img

def apply_clahe(img):
    """应用 CLAHE 对比度增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img

def process_single_image(args):
    """单张图像的完整处理管线"""
    img_id, src_dir, dst_dir, target_size = args
    img_path = os.path.join(src_dir, f"{img_id}.png")
    save_path = os.path.join(dst_dir, f"{img_id}.png")
    
    # 如果文件已经存在，直接跳过 (方便中断后继续)
    if os.path.exists(save_path):
        return True
        
    image = cv2.imread(img_path)
    if image is None:
        return False
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 核心预处理流水线
    image = crop_image_from_gray(image)      # 1. 裁边
    image = circle_crop(image)               # 2. 圆形 ROI 提取
    image = cv2.resize(image, (target_size, target_size))  # 3. 缩放
    image = apply_clahe(image)               # 4. 在标准尺寸上应用增强
    
    # 保存前转回 BGR 格式给 cv2 写入
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
    return True

def main():
    csv_path = 'data/raw/train.csv'
    src_dir = 'data/raw/train_images'
    dst_dir = 'data/processed/train_images'
    target_size = 384
    
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    img_ids = df['id_code'].tolist()
    
    # 构造多进程参数
    tasks = [(img_id, src_dir, dst_dir, target_size) for img_id in img_ids]
    
    print(f"离线预处理 {len(tasks)} 张图片...")
    # 开启 8 个进程加速处理 (可根据 CPU 核心数调整)
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Processing"))
        
    print(f"\n预处理完成！全部图片已保存至 {dst_dir}")

if __name__ == '__main__':
    main()
