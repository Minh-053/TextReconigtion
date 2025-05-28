import pytesseract
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import pandas as pd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import sys
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
sys.stdout.reconfigure(encoding='utf-8')

# ======= BƯỚC 1: Load ảnh ========
image_path = r'G:\Text Recognition 1\image\7.png'
image = Image.open(image_path)

# ======= HÀM TIỀN XỬ LÝ ẢNH ========
def preprocess_image(img):
    # 1. Chuyển sang grayscale
    img = img.convert('L')

    # 2. Tăng tương phản mạnh
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)  

    # 3. Làm nét ảnh
    img = img.filter(ImageFilter.SHARPEN)

    # 4. Nhị phân hóa ảnh (thresholding)
    threshold = 160  
    img = img.point(lambda x: 255 if x > threshold else 0, mode='1')  # ảnh nhị phân

    # 5. Resize chiều cao 
    target_height = 128
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    img = img.resize((new_width, target_height))

    return img

# ======= BƯỚC 2: Dùng Tesseract để phát hiện từ ========
tsv_data = pytesseract.image_to_data(image, lang='vie', output_type=pytesseract.Output.DATAFRAME)
tsv_data = tsv_data[tsv_data['text'].notnull()]
tsv_data = tsv_data[tsv_data['conf'] != '-1']

lines = []
grouped = tsv_data.groupby(['block_num', 'par_num', 'line_num'])

for (block, para, line), line_df in grouped:
    line_df = line_df.sort_values(by='left')
    lines.append(line_df)

# ======= BƯỚC 3: Load mô hình VietOCR ========
config = Cfg.load_config_from_name('vgg_seq2seq')
config['weights'] = r'G:\Text Recognition 1\weight\seq2seqocr1.pth'
config['device'] = 'cuda'  
config['predictor']['beamsearch'] = False
ocr_model = Predictor(config)

# ======= BƯỚC 4: Chạy OCR từng dòng ========
recognized_sentences = []

for line_df in lines:
    if line_df.empty:
        continue

    x1 = line_df['left'].min()
    y1 = line_df['top'].min()
    x2 = (line_df['left'] + line_df['width']).max()
    y2 = (line_df['top'] + line_df['height']).max()

    cropped_line_img = image.crop((x1, y1, x2, y2))

    try:
        processed_img = preprocess_image(cropped_line_img)
        sentence = ocr_model.predict(processed_img)
        recognized_sentences.append(sentence)
    except Exception as e:
        print(f"Lỗi nhận diện tại dòng ({x1},{y1},{x2},{y2}): {e}")

# ======= BƯỚC 5: Hiển thị ảnh và văn bản trong popup ========
def show_result(image, text):
    window = tk.Tk()
    window.title("Kết quả nhận diện văn bản")
    window.geometry("800x600")

    img = image.copy()
    img.thumbnail((600, 400))
    tk_img = ImageTk.PhotoImage(img)
    img_label = tk.Label(window, image=tk_img)
    img_label.image = tk_img
    img_label.pack(pady=10)

    textbox = ScrolledText(window, wrap=tk.WORD, font=("Arial", 12))
    textbox.pack(fill=tk.BOTH, expand=True)
    textbox.insert(tk.END, text)
    textbox.configure(state='disabled')

    window.mainloop()

# ======= In kết quả ========
final_text = " ".join(recognized_sentences)
show_result(image, final_text)
