# Module phát hiện vật thể theo màu bằng webcam

Module độc lập giao cho Người 3 tích hợp GUI. Ảnh đầu vào/đầu ra dùng BGR `uint8` theo OpenCV. Quy trình: Gaussian blur → BGR sang HSV → mask 6 màu → opening/closing → contour ngoài → lọc diện tích → bbox, tâm, nhãn và thống kê.

## Cài đặt

Python 3.9 trở lên:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## API tích hợp GUI

```python
from color_detector import open_config, process_frame, process_image_path

config = open_config("config/colors.json")
result_bgr, stats = process_image_path("data/images/mau_thu.jpg", config)
# Hoặc với frame webcam BGR vừa đọc từ VideoCapture:
# result_bgr, stats = process_frame(frame_bgr, config)
print(stats["counts"], stats["total"])
for obj in stats["objects"]:
    print(obj["index"], obj["label"], obj["bbox"], obj["center"], obj["area"])
```

Định dạng trả về:

- `result_bgr`: ảnh BGR đã vẽ hộp bao, tâm, nhãn màu và tổng số.
- `stats["counts"]`: số lượng theo key màu (`red`, `orange`, `yellow`, `green`, `blue`, `purple`).
- `stats["total"]`: tổng vật thể được nhận diện.
- `stats["objects"]`: danh sách `{index, color, label, bbox, center, area}`; `bbox=(x,y,w,h)`, `center=(cx,cy)` theo pixel ảnh gốc.

Ảnh tĩnh lỗi sẽ báo `ValueError`; ảnh frame sai định dạng báo `ValueError`/`TypeError`. `process_frame(..., return_masks=True)` trả thêm mask đã morphology trong `stats["masks"]` để xem/debug. `save_result(path, result_bgr)` lưu ảnh và báo lỗi nếu ghi thất bại.

## Webcam

```python
from color_detector import open_camera, process_frame, open_config

cap = open_camera(0)  # ném RuntimeError nếu không mở được camera
config = open_config("config/colors.json")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result, stats = process_frame(frame, config)
        # GUI chuyển result BGR sang RGB trước khi đưa cho Tkinter/Pillow.
finally:
    cap.release()  # Người 3 cần gọi khi tắt webcam/đóng app
```

## Hiệu chỉnh

Sửa `config/colors.json`, không cần sửa thuật toán. HSV dùng thang OpenCV: H 0–179, S/V 0–255. Mỗi màu nhận một hay nhiều cặp cận `[lower, upper]`; đỏ có hai khoảng Hue. `min_area=500` lấy theo ví dụ trong báo cáo, cần giảm khi kiểm tra ảnh vật thể nhỏ hoặc tăng để bỏ nhiễu. Kernel morphology nên là số lẻ; cấu hình số chẵn sẽ tự tăng lên số lẻ gần nhất. Bật/tắt màu bằng `enabled_colors`.

## Dữ liệu và kiểm thử

ZIP ảnh mẫu hiện có 16 ảnh đã rà soát: ảnh 01–07 và 16 là ảnh minh họa tổng hợp; 08–14 là ảnh thực tế; 15 là ảnh tham chiếu, không tính ảnh phát hiện vật thể. Bộ này không chứa IMG01–IMG30 và nhãn `labels.csv` trong phụ lục báo cáo. Vì vậy chưa có căn cứ ghi Pass/Fail cho TC01–TC12; cần chạy từng ca trên ảnh có nhãn thực tế và ghi nhận kết quả, đặc biệt các ca nền phức tạp, ánh sáng yếu/chói, nền gần màu và vật thể nhỏ.

## Giới hạn đã biết

Phân ngưỡng dựa trên HSV nên kết quả phụ thuộc ánh sáng và nền. Contour ngoài sẽ gộp hai vật thể cùng màu nếu mask của chúng dính nhau. Thống kê là kết quả thuật toán, không thay cho nhãn thực tế/đo độ chính xác. FPS nên được đo ở vòng webcam của ứng dụng tích hợp.
