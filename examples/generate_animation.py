import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from datetime import datetime
import traceback
from tqdm import tqdm
from PIL import Image

def resize_image(img, target_size=(1200, 900)):
    """调整图片尺寸"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return np.array(img.resize(target_size, Image.Resampling.LANCZOS))

def create_visualization(data_dir: str):
    """从保存的数据创建可视化"""
    try:
        print(f"Using data from: {data_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")
        
        # 创建输出目录
        output_dir = os.path.join('results', 'animations', 
                                 datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有PNG文件
        image_files = sorted([f for f in os.listdir(data_dir) 
                            if f.endswith('.png')])
        
        if not image_files:
            raise ValueError(f"No PNG files found in {data_dir}")
            
        print(f"Found {len(image_files)} images")
        
        # 读取第一张图片来获取目标尺寸
        first_img = Image.open(os.path.join(data_dir, image_files[0]))
        target_size = first_img.size
        print(f"Target image size: {target_size}")
        
        # 读取并调整所有图像
        images = []
        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                img_path = os.path.join(data_dir, img_file)
                img = Image.open(img_path)
                
                # 如果尺寸不同，进行调整
                if img.size != target_size:
                    print(f"Resizing image {img_file} from {img.size} to {target_size}")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # 转换为numpy数组
                img_array = np.array(img)
                images.append(img_array)
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        if not images:
            raise ValueError("No valid images could be loaded!")
        
        # 验证所有图片尺寸
        shapes = [img.shape for img in images]
        if len(set(str(shape) for shape in shapes)) > 1:
            print("Warning: Images have different shapes:", set(str(shape) for shape in shapes))
            raise ValueError("Images must have the same shape")
        
        print("Generating GIF...")
        # 生成GIF
        output_path = os.path.join(output_dir, 'simulation.gif')
        imageio.mimsave(output_path, images, duration=0.2, loop=0)
        print(f"GIF saved to {output_path}")
        
        print("Generating MP4...")
        # 生成MP4
        try:
            import moviepy.editor as mpy
            clip = mpy.ImageSequenceClip(images, fps=10)
            mp4_path = os.path.join(output_dir, 'simulation.mp4')
            clip.write_videofile(mp4_path, fps=10, verbose=False, logger=None)
            print(f"MP4 saved to {mp4_path}")
        except Exception as e:
            print(f"Could not create MP4: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")
        traceback.print_exc()

def main():
    try:
        # 使用绝对路径
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(current_dir, 'results', 'visualizations', '20241209_165100')
        
        print(f"Looking for data in: {data_dir}")
        create_visualization(data_dir)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 