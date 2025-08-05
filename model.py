from ultralytics import YOLO
import torch


def main():
    print("CUDA Available:", torch.cuda.is_available())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    print(torch.__version__)
    print(torch.version.cuda)

    model = YOLO("yolov8n.pt")
    model.train(data="Fridge detection.v1i.yolov8/data.yaml", epochs=50, imgsz=640)

if __name__ == "__main__":
    main()