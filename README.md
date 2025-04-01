# Rice Grain Recognition

🚀 **Hệ thống nhận diện và phân loại hạt gạo bằng AI**

## 🔹 Giới thiệu
Dự án này sử dụng **Deep Learning** để nhận diện và phân loại các loại hạt gạo dựa trên hình ảnh. Hệ thống có thể hỗ trợ đánh giá chất lượng gạo và ứng dụng trong chuỗi cung ứng thực phẩm.

## 📌 Dữ liệu
- **Nguồn dữ liệu**: [Rice Image Dataset - Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)
- **Bao gồm 5 loại gạo**:
  - `Arborio`
  - `Basmati`
  - `Ipsala`
  - `Jasmine`
  - `Karacadag`

## 🛠️ Công nghệ sử dụng
- **Python**, **Flask** (Backend API)
- **TensorFlow/Keras** hoặc **PyTorch** (Deep Learning)
- **OpenCV** (Xử lý ảnh)

## 🚀 Chức năng chính
✅ **Nhận diện hạt gạo từ hình ảnh**

## 📌 Hướng dẫn sử dụng
### 1️⃣ Clone repository
```bash
git clone https://github.com/thangdac/Rice-Grain-Recognition.git
cd Rice-Grain-Recognition
```xx

### 2️⃣ Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3️⃣ Chạy API Flask
```bash
python app.py
```

### 4️⃣ Gửi request nhận diện gạo
```bash
curl -X POST -F "file=@test.jpg" http://localhost:11223/predict
```

## 📊 Kết quả
Mô hình có độ chính xác cao và có thể mở rộng để nhận diện thêm nhiều loại gạo khác nhau.

---

📧 **Liên hệ:** Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ qua email hoặc mở issue trên GitHub! 🚀