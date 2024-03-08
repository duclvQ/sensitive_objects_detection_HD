from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO(r"E:\HD_VNese_map\main_src\yolov8n\runs\detect\train41\weights\last.pt")  # load a pretrained model (recommended for training)
if __name__ == '__main__':
    # Use the model
    model.train(data="dataset.yaml",resume=True, epochs=200, batch=8, pretrained=True)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="torchscript")  # export the model to ONNX format