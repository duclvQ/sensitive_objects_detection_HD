from ultralytics import YOLO

# Load a model
model = YOLO("yolov6m.yaml")  # build a new model from scratch
#model = YOLO("yolov6m.pt")  # load a pretrained model (recommended for training)
if __name__ == '__main__':
    # Use the model
    model.train(data="dataset.yaml", epochs=50, batch=20, pretrained=True, lr0=0.01)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format