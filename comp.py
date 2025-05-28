import os
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import Levenshtein
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ===== Load Ground Truth =====
gt_path = r'G:\Text Recognition 1\test\test.txt' #file text.txt
image_folder = r'G:\Text Recognition 1\test' #folder test
ground_truth = {}

with open(gt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if '\t' in line:
            filename, gt_text = line.strip().split('\t')
            ground_truth[filename] = gt_text.strip()

# ===== Hàm tính CER =====
def cer(s1, s2):
    return Levenshtein.distance(s1, s2) / max(len(s2), 1)

# ===== Hàm đánh giá mô hình =====
def evaluate_model(model: Predictor, ground_truth):
    total_cer = 0
    total = 0
    results = []

    for filename, gt_text in ground_truth.items():
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path)

        try:
            pred_text = model.predict(image).strip()
        except Exception as e:
            pred_text = ''
            print(f'Lỗi nhận diện {filename}: {e}')

        error = cer(pred_text, gt_text)
        total_cer += error
        total += 1
        results.append((filename, gt_text, pred_text, error))

    avg_cer = total_cer / total if total else 0
    return avg_cer, results

# ===== Load mô hình gốc =====
def load_vietocr_model(weights=None):
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cuda'  
    config['predictor']['beamsearch'] = True
    if weights:
        config['weights'] = weights
    return Predictor(config)

# ===== So sánh 2 mô hình =====
# Mô hình gốc
model_pretrained = load_vietocr_model()
cer_pretrained, res_pretrained = evaluate_model(model_pretrained, ground_truth)

# Mô hình tự huấn luyện
model_custom = load_vietocr_model(weights='G:/Text Recognition 1/weight/seq2seqocr1.pth')
cer_custom, res_custom = evaluate_model(model_custom, ground_truth)

# ===== In kết quả so sánh =====
print(f"\n CER mô hình gốc: {cer_pretrained:.4f}")
print(f" CER mô hình train: {cer_custom:.4f}\n")

for i, (filename, gt, pred_custom, err_custom) in enumerate(res_custom):
    pred_pretrained = res_pretrained[i][2]
    print(f"[{filename}]")
    print(f"GT         : {gt}")
    print(f"Original : {pred_pretrained}")
    print(f"Trained     : {pred_custom}")
    print(f"CER Trained : {err_custom:.4f}")
    print("-" * 50)
