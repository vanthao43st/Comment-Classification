# Comment Classification (PhoBERT finetune)

Dự án gồm 2 phần chính:

1. training/ → dùng để huấn luyện model (chạy local hoặc Kaggle)
2. serving/ → dùng để triển khai API Flask + Ngrok

==========================================</br>
I. HUẤN LUYỆN MÔ HÌNH (TRAINING)
==========================================

## Cách 1: Chạy Jupyter Notebook trên local

1. Mở thư mục training:
   cd training

2. Mở file notebook:
   jupyter notebook Phobert_Finetune_Trainer.ipynb

3. Notebook sẽ tự nhận môi trường local và lưu model sau khi train vào:
   ../serving/models/phobert_sentiment/

=> Khi xong, chỉ cần sang thư mục serving/ là có thể chạy Flask ngay.

---

## Cách 2: Chạy notebook trên Kaggle

1. Upload toàn bộ thư mục training/ lên Kaggle.

2. Mở file Phobert_Finetune_Trainer.ipynb và chạy.

3. Notebook sẽ tự nhận môi trường Kaggle, lưu model tại:
   /kaggle/working/models/phobert_sentiment/

4. Sau khi train xong, notebook sẽ tự nén thành:
   /kaggle/working/models.zip

5. Tải file models.zip về, giải nén, sau đó copy thư mục:
   phobert_sentiment/
   → vào thư mục: serving/models/phobert_sentiment/

==========================================</br>
II. TRIỂN KHAI MÔ HÌNH (SERVING)
==========================================

1. Di chuyển sang thư mục serving:
   cd serving

2. Cài các thư viện cần thiết:
   pip install -r requirements.txt

3. Tạo file .env và thêm dòng:
   NGROK_KEY=your_ngrok_auth_token_here

4. Chạy Flask server:
   python app.py

Khi chạy xong sẽ thấy:
Ngrok URL: https://xxxxx.ngrok.io
Flask running on http://0.0.0.0:5000

=> Dán URL ngrok vào Postman hoặc trình duyệt để test API.

---

## III. KIỂM TRA API

Cách 1: Gửi request bằng curl</br>
curl -X POST http://127.0.0.1:5000/predict \</br>
-H "Content-Type: application/json" \</br>
-d '{"text": "Sản phẩm rất tốt, giao hàng nhanh"}'</br>

Cách 2: Gửi request bằng Python</br>
import requests</br>
res = requests.post("http://127.0.0.1:5000/predict", json={"text": "Dịch vụ tệ"})</br>
print(res.json())</br>

Kết quả mẫu:</br>

```json
{
  "success": true,
  "result": {
    "original_text": "Dịch vụ tệ",
    "cleaned_text": "Dịch vụ tệ",
    "segmented_text": "Dịch_vụ tệ",
    "predicted_class": "Tiêu cực",
    "probabilities": {
      "Tiêu cực": 0.85,
      "Trung lập": 0.1,
      "Tích cực": 0.05
    }
  }
}
```

## IV. TÓM TẮT QUY TRÌNH

- Local:

  - Chạy notebook hoặc train_phobert.py trong thư mục training
  - Model tự lưu sang serving/models/phobert_sentiment/
  - Sang serving và chạy python app.py

- Kaggle:
  - Chạy notebook Phobert_Finetune_Trainer.ipynb
  - Tải models.zip về, giải nén
  - Copy thư mục phobert_sentiment vào serving/models/
  - Chạy python app.py

---

## V. THÔNG TIN DỮ LIỆU

File CSV trong training/dataset cần có 2 cột:
comment_text, labels

Ví dụ:</br>
Sản phẩm rất tốt,2</br>
Dịch vụ tệ,0</br>
Bình thường,1</br>

Nhãn cảm xúc:
0 = Tiêu cực
1 = Trung lập
2 = Tích cực
