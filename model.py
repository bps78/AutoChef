from ultralytics import YOLO
import torch
import os


def main():
    print("CUDA Available:", torch.cuda.is_available())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    print(torch.__version__)
    print(torch.version.cuda)


    os.environ["YOLO_SAVE_DIR"] = "C:/Users/bps78/Documents/GitHub/AutoDJ/runs/train"
    model = YOLO("yolov8n.pt")
    model.train(data="fridgeDataV2/data.yaml", epochs=60, imgsz=640, save_dir="C:/Users/bps78/Documents/GitHub/AutoDJ/runs/train")

if __name__ == "__main__":
    main()