from data_utils import *

class COCO_json(object):
    """
    Class that allows to create COCO-format annotations for an arbitrary number of images.

    Inputs:
        images_dir : str
            Path to where the images are stored
        meta_dir : str
            Path to meta_dir for the dataset
         categories_dict : dict
            Dictionary that maps category names to category ids
        sets : list or string
            List or string that names the set that will be used to extract annotations from
            it should be ['train'] / ['test'] or ['train', 'test']
        images_paths_list : list
            List of absolute paths that points to images from the dataset
    """
    def __init__(self, images_dir, meta_dir, categories_dict, sets, images_paths_list):
        self.images_dir = images_dir
        self.meta_dir = meta_dir
        self.categories_dict = categories_dict
        self.sets = sets
        self.images_paths_list = images_paths_list

        def check_ext():
            if 'txt' in self.images_paths_list[0]:
                self.images_paths_list = [x.replace('txt', 'jpg') for x in self.images_paths_list]
        
        if self.images_paths_list is not None:
            check_ext()

    def create_info(self, year=2019, version=1.0, desc='', contr='', url='', datetime=''):
        info = {
            "year": year,
            "version": version,
            "description": desc,
            "contributor": contr,
            "url": url,
            "date_created": datetime
        }

        return info

    def create_license(self, idx=0, name='', url=''):
        license = {
            "id": idx,
            "name": name,
            "url": url
        }

        return license

    def create_image_info(self, image_id, width, height, file_name, license=0, flickr_url='', coco_url='', data_captured=''):
        image = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": license,
            "flickr_url": flickr_url,
            "coco_url": coco_url,
            "date_captured": data_captured
        }

        return image

    def create_images_info_all(self):

        self.images = []
        for idx, image_name in enumerate(self.images_paths_list):
            image_path = os.path.join(self.images_dir, image_name)
            width, height = get_images_size(image_path)
            img_id = get_image_id(image_name)
            self.images.append(self.create_image_info(image_id=img_id, width=width, height=height, file_name=image_name))

    def create_annotations(self, anno_id, image_id, category_id, bbox, segmentation='', area='', iscrowd=0):
        annotation = {
            "id": anno_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd
        }

        return annotation

    def create_categories(self, category_id, category_name, supercategory='fashion'):
        categories = {
            "id": category_id,
            "name": category_name,
            "supercategory": supercategory
        }

        return categories

    def create_annotations_all(self, bbox_transform_func=None):

        # All files with street photos will be used for training
        if not type(self.sets) == list:
            self.sets = list(self.sets)

        anno_id = 0
        self.annotations = []
        self.categories = []
        for sett in self.sets:
            for category in list(self.categories_dict.keys()):
                filepath = os.path.join(self.meta_dir, f'{sett}_pairs_{category}.json')

                json_file = json.load(open(filepath))

                if not len(self.categories) == len(list(self.categories_dict.keys())):
                    category_name = self.categories_dict[category][0]
                    category_id = self.categories_dict[category][1]
                    # print(f'category_name, category_id | {category_name}, {category_id}')
                    self.categories.append(self.create_categories(category_id=category_id, category_name=category_name, supercategory='fashion'))

                for anno in json_file:
                    image_id = anno['photo']
                    if bbox_transform_func is not None:
                        bbox = bbox_transform_func(bbox=anno['bbox'])
                    else:
                        bbox = bbox=anno['bbox']
                    self.annotations.append(self.create_annotations(anno_id=anno_id, image_id=image_id, category_id=category_id, bbox=bbox, segmentation='', area='', iscrowd=0))

                    anno_id += 1

    def create_full_coco_json(self, bbox_transform_func=None):

        self.info = self.create_info()
        self.licenses = self.create_license()
        self.create_images_info_all()
        self.create_annotations_all(bbox_transform_func=bbox_transform_func)

        self.json = {
            'info': self.info,
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories,
            'licenses': self.licenses
        }

