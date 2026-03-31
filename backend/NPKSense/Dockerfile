FROM python:3.12-slim

# เปลี่ยนชื่อแพ็กเกจให้รองรับ Debian รุ่นใหม่
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy เฉพาะไฟล์ที่จำเป็น
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy ไฟล์โค้ดและโมเดลทั้งหมด
COPY --chown=user . .

# สั่งรันพอร์ต 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]