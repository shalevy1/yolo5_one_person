import cv2
import os
from pathlib import Path
import numpy as np
import json

def create_crop_save_path(crop_save_dir:Path, *args):
    ext='jpg'
    args = [str(arg) for arg in args]
    fname = '_'.join(args)
    fname = fname + '.' + ext
    return str(crop_save_dir / fname)

def crop_n_save(xyxy, img:np.ndarray, crop_save_path:str):
    x1,y1,x2,y2 = xyxy
    cv2.imwrite(crop_save_path, img[int(y1):int(y2), int(x1):int(x2)])

class JsonWriter(object):

    def __init__(self, save_dir:str, fname:str='cropped_detections.json'):
        self.save_dir = save_dir
        self.fname = fname
        self.save_path = os.path.join(self.save_dir, self.fname)
        self.json = []
        self.id = 0

    def add_item(self, **kwargs):
        """
        kwargs: det_fname:str, frame_num:int, bbox:list, cls:int
        """
        item = self.create_entry(**kwargs)
        self.json.append(item)

    def create_entry(self, file_name:str, frame_num:int, bbox:list, cls:int):
        id = self.id
        self.id += 1
        entry = {'id':id, 'file_name':file_name, 'frame_num':frame_num, 'bbox':bbox, 'class_id':cls} 
        return entry

    def save_json(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.json, f)