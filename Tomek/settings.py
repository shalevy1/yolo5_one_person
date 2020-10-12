class Settings():

    def __init__(self):
        print("tt")
        self.weights = './weights/best.pt'
        self.source = "rtsp://admin:12345678-q@192.168.1.70:554"
        self.output ='inference/output'
        self.img_size = 640
        self.save_json = False
        self.crop_detections = False
        self.view_img = True
        self.save_txt = False #True
        self.device = "cpu"
        self.agnostic_nms=False
        self.augment = False
        self.classes = [0]
        self.iou_thres = 0.5
        self.update = False
        self.conf_thres = 0.4