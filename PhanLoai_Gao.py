import torch
from torchvision import transforms
from PIL import Image
import cv2
import torch.nn as nn
import torch.nn.functional as F 
from ultralytics import YOLO
from flask_cors import CORS  
from flask import Flask, request, jsonify, send_file
import numpy as np
import io
import base64


app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho toàn bộ ứng dụng

# Tải mô hình YOLO
# model_YOLO = YOLO('yolov8n.pt')


# Định nghĩa các lớp gạo (theo thứ tự nhãn mà bạn đã huấn luyện)
Classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"] 

# Đường dẫn đến file mô hình
model_path = "rice_classification_model.pth" 
yolo_model_path = "C:/Users/Admin/Documents/DACN/Rice_Model/nhandienlor/runs/detect/train4/weights/best.pt"  # Mô hình YOLOv8 đã huấn luyện

yolo_model = YOLO(yolo_model_path)  

# Tải mô hình từ file .pth
def load_model(model_path):
    model.load_state_dict(torch.load(model_path))  # Tải trọng số vào mô hình
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    return model

#chọn thiết bị train
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#xây dựng mô hình cnn trong pytorch
class CNN(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels = 3 , out_channels = 32 , kernel_size = (3,3) , stride = 1), # 64 conv filter
            nn.BatchNorm2d(32), # Normalize the outputs of the previous layer
            nn.ReLU(), # Apply ReLU activation
            nn.MaxPool2d(kernel_size = (2,2) , stride = 2), # Reduce spatial dimensions by half
            # ُSecond conv layer
            nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = (3,3) , stride = 1), # 64 conv filter
            nn.BatchNorm2d(64), # Normalize the outputs of the previous layer
            nn.ReLU(), # Apply ReLU activation
            nn.BatchNorm2d(64), # Normalize the outputs of the previous layer
            nn.Conv2d(in_channels = 64 , out_channels = 64 , kernel_size = (3,3) , stride = 1), # 64 conv filter
            nn.ReLU(), # Apply ReLU activation
            nn.MaxPool2d(kernel_size = (2,2) , stride = 2), # Reduce spatial dimensions by half
            # Third conv layer
            nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size = (3,3) , stride = 1), # 128 conv filter
            nn.BatchNorm2d(128), # Normalize the outputs of the previous layer
            nn.ReLU(), # Apply ReLU activation
            nn.Conv2d(in_channels = 128 , out_channels = 128 , kernel_size = (3,3) , stride = 1), # 128 conv filter
            nn.BatchNorm2d(128), # Normalize the outputs of the previous layer
            nn.ReLU(), # Apply ReLU activation
            nn.MaxPool2d(kernel_size = (2,2) , stride = 2) # Reduce spatial dimensions by half
        )
        self.fully_connected = nn.Sequential(
            # Fully conected layer
            nn.Linear(73728 , 512), # Fully connected layer with 512 node
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout layer for regularization
            nn.Linear(512 , 5) # Final layer: From 512 units to the number of classes
        )
    def forward(self , X) :
        out = self.conv_layers(X)
        out = torch.flatten(out , 1)
        out = self.fully_connected(out)
        return out # Return the final output (logits or class scores)
model = CNN()

#--------------------------------------------------------------------------------------

# Định nghĩa hàm tiền xử lý ảnh
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image


#Hàm dự đoán loại gạo từ ảnh
def predict_rice_type(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
    # Kiểm tra xác suất tối đa trước khi đưa ra dự đoán
    if max_prob.item() < 0.8:  # Ngưỡng xác suất
        return "không nhận diện được hạt gạo!!", max_prob.item()
    return Classes[predicted_class.item()], max_prob.item()
 

# Endpoint để nhận hình ảnh và trả về loại gạo
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Tiền xử lý ảnh
        image = Image.open(file).convert('RGB')
        image_tensor = preprocess_image(image)

        #------phương án 1 ------------- 
        # Dự đoán loại gạo
        rice_type = predict_rice_type(image_tensor, model, device)
        return jsonify({"riceType": rice_type})

        #------phương án 2 ------------- 
        # if len(results.pred[0]) == 0:
        #     # Nếu không có vật thể nào nhận diện được, tiếp tục dự đoán loại gạo
        #     image_tensor = preprocess_image(image)
        #     riceType = predict_rice_type(image_tensor, model, device)
        #     return jsonify({"riceType": riceType})
        # else:
        #     # Nếu có vật thể nhận diện được, hiển thị kết quả nhận diện
        #     detectedObjects = []
        #     predicted_labels = results.names  # Các tên lớp dự đoán
        #     for idx, pred in enumerate(results.pred[0]):
        #         label = predicted_labels[int(pred[5])]  # Tên lớp dựa trên chỉ số lớp
        #         detectedObjects.append(label)
        #     return jsonify({"detectedObjects": detectedObjects})
        
        #------phương án 3 ------------- 
        # detectedObjects = []
        # predicted_labels = results.names  # Các tên lớp dự đoán
        # rice_detected = False  # Biến kiểm tra xem có "rice" không
        # # Lặp qua các vật thể nhận diện và kiểm tra xem có phải là gạo không
        # for idx, pred in enumerate(results.pred[0]):
        #     label = predicted_labels[int(pred[5])]  # Tên lớp dựa trên chỉ số lớp
        #     detectedObjects.append(label)
            
        #     if label.lower() == 'rice':  # Kiểm tra xem có phải "rice" không (case-insensitive)
        #         rice_detected = True

        # if rice_detected:
        #     image_tensor = preprocess_image(image)
        #     riceType = predict_rice_type(image_tensor, model, device)
        #     return jsonify({"riceType": riceType})
        # if len(results.pred[0]) == 0:
        #     return jsonify({"message": "Không nhận diện được vật thể."})
        # else:
        #     return jsonify({"detectedObjects": detectedObjects})

    #------------------------------------------------------ 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra nếu ảnh được gửi kèm theo yêu cầu
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # Chuyển ảnh từ PIL.Image sang numpy.ndarray
    image_np = np.array(image)

    # Dự đoán các đối tượng trong ảnh bằng YOLO
    results = yolo_model(image)
    if len(results[0].boxes) == 0:
        return jsonify({"error": "No object detected"}), 400

    boxes = results[0].boxes.data.cpu().numpy()
    predictions = []
    
    # Xử lý các bounding boxes
    for i, box in enumerate(boxes):
        if len(box) >= 5:
            x_min, y_min, x_max, y_max, confidence = box[:5]
            if confidence < 0.3:
                continue

            # Tinh chỉnh vùng cắt thông minh
            x_min = max(0, int(x_min) - 50)
            y_min = max(0, int(y_min) - 50)
            x_max = min(image.width, int(x_max) + 50)
            y_max = min(image.height, int(y_max) + 50)

            grain_roi = image_np[y_min:y_max, x_min:x_max]
            grain_resized = cv2.resize(grain_roi, (224, 224))
            grain_rgb = cv2.cvtColor(grain_resized, cv2.COLOR_BGR2RGB)
            grain_img = Image.fromarray(grain_rgb)

            grain_tensor = preprocess_image(grain_img).to(device)

            # Dự đoán loại gạo
            rice_type, prob = predict_rice_type(grain_tensor, model, device)

            # Vẽ bounding box và hiển thị loại gạo
            cv2.rectangle(image_np, (x_min + 50, y_min + 50), (x_max - 50, y_max - 50), (0, 255, 0), 2)
            cv2.putText(
                image_np,
                f"{rice_type} ({prob:.2f})",
                (x_min + 50, y_min + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            predictions.append({
                "grain": i + 1,
                "rice_type": rice_type,
                "probability": prob
            })

    _, img_bytes = cv2.imencode('.png', image_np)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Trả về JSON chứa ảnh và thông tin loại gạo
    response = {
        "image": img_base64,
        "predictions": predictions
    }
    return jsonify(response)

if __name__ == '__main__':
    load_model(model_path)
    app.run(host="0.0.0.0", port=11223)
