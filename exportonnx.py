from ultralytics import YOLO

model_path="./yolov8s.pt"
# Load a model
model = YOLO(model_path)  # load a custom trained

# Export the model
model.export(format='onnx')