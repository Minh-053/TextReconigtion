import cv2
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

input_folder = "G:\Text Recognition 1\image"
output_folder = "G:\\Text Recognition 1\\dataset\\val"
os.makedirs(output_folder, exist_ok=True)

# Tăng tương phản bằng CLAHE
def enhance_contrast(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

for image_file in os.listdir(input_folder):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠ Không đọc được ảnh: {image_file}")
        continue

    # ==== TIỀN XỬ LÝ NÂNG CẤP ====
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)

    enhanced = enhance_contrast(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )

    # Ghi ảnh (dùng PNG để giữ chi tiết)
    output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".png")
    cv2.imwrite(output_path, binary)
    print(f"✅ Xử lý xong: {output_path}")

print("🎯 Tiền xử lý nâng cấp hoàn tất!")
