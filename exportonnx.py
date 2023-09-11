

from ultralytics import YOLO

model = YOLO("best.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx")  # export the model to ONNX format