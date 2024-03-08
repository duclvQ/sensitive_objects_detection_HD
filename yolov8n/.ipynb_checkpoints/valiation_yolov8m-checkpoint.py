


from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train7/weights/best.pt")  # load a pretrained model (recommended for training)
if __name__ == '__main__':
    # Use the model
    #model.train(data="dataset.yaml", epochs=20, batch=32, pretrained=True, lr0=0.01)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format