import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def crop_to_square_roi(image, tol=7):
    """
    提取眼底区域并裁剪为正方形。
    通过找到非纯黑像素的边界框，并取最大边长作为正方形的边，
    以保证在后续 resize 时不会发生拉伸形变。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    
    # 获取非黑像素的坐标
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image # 如果全黑则直接返回原图，容错处理
        
    # 获取紧凑边界框
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 计算边界框的中心点和最大边长
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    max_size = max(y_max - y_min, x_max - x_min)
    half_size = max_size // 2
    
    # 计算正方形裁剪区域的坐标
    y1 = center_y - half_size
    y2 = center_y + half_size
    x1 = center_x - half_size
    x2 = center_x + half_size
    
    # 如果完美正方形超出了原图范围，需要用黑色填充（Padding）
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - image.shape[0])
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - image.shape[1])
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # 调整裁剪坐标适应填充后的新图
        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left
        
    # 截出完美的正方形
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def circle_crop(img):
    """圆形 ROI 裁剪，去除边缘光晕和暗角干扰"""
    height, width, depth = img.shape
    x = width // 2
    y = height // 2
    r = min(x, y)
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    
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
    
    if os.path.exists(save_path):
        return True
        
    image = cv2.imread(img_path)
    if image is None:
        return False
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 核心预处理流水线
    image = crop_to_square_roi(image)
    image = circle_crop(image)
    
    # 3. 此时图已是正方形，直接 resize 不会有任何比例形变
    image = cv2.resize(image, (target_size, target_size))  
    image = apply_clahe(image)
    
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
    
    tasks = [(img_id, src_dir, dst_dir, target_size) for img_id in img_ids]
    
    print(f"离线预处理 {len(tasks)} 张图片...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Processing"))
        
    print(f"\n预处理完成！全部图片已保存至 {dst_dir}")

if __name__ == '__main__':
    main()
