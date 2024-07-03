from ultralytics import YOLO

# load model
model = YOLO('C:/Users/bigfatcat\Desktop/zed-yolo/YOLOv8-stereo-main/ultralytics-main/yolov8n.pt')

# Train the model
model.train(data = 'C:/Users/bigfatcat/Desktop/zed-yolo/YOLOv8-stereo-main/ultralytics-main/yolo_digger.yaml', workers = 0, epochs = 500, batch = 16)