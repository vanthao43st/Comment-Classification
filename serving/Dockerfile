# 1) Chọn base image nhỏ, đủ dùng
FROM python:3.10-slim

# 2) Biến môi trường Python "gọn sạch"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3) Thư mục làm việc trong container
WORKDIR /app

# 4) Copy riêng requirements để tận dụng layer cache
COPY requirements.txt /app/requirements.txt

# 5) Cài gói hệ thống tối thiểu (nếu cần cho wheels), rồi cài pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# (Nếu torch cài chậm/lỗi trên slim, dùng CPU wheel chính chủ:
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu )

# 6) Copy code (và có thể cả models) vào image
COPY . /app

# 7) Cổng dịch vụ
EXPOSE 8080

# 8) Lệnh chạy: dùng gunicorn (app.py có biến app = Flask(...))
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]