import os
import numpy as np
from data_utils import *
from data_format_transforms import *
from coco_format_utils import COCO_json

from tqdm import tqdm

#TODO Modify the pipeline to be inline with Yolov5 requirments
def transform_annotations_to_yolo(json_data, images_data, images_data_path):
    # Remove all txt file so that to avoid repeating annotation for the same image
    remove_all_annotation_files(path=images_data_path, ext='txt')

    prev_img_id = -1  # Variable to not to search for width and height of the same image every loop

    for anno in tqdm(json_data["annotations"]):
        # print(anno)
        img_id = anno['image_id']
        category = anno['category_id']
        bbox = anno['bbox'].copy()
        #     img_match_id = json_data["annotations"][idx]['id']
        #     print(bbox)
        if prev_img_id != img_id:
            # Loop over images to find the image corressponding to
            # annotation and get its' height and width
            # As there can be multiple annotations for one image
            # this step is performed only once, for the first time for and image.
            width, height, _ = search_for_image_dimension(images_data=images_data, identifier='id', img_searched=img_id)
            prev_img_id = img_id
        #                 print('New image fetched')
        # At this stage bbox are [top_left_x, top_left_y, width, height]
        bbox = bbox_coco_to_yolo(bbox, width, height)
        # At this stage bbox are [x_center_of_bbox, y_center_of_bbox, width, height]
        # all in terms of relative dimension of an image == YOLO FORMAT

        #     print(bbox)
        img_id = str(img_id)
        txt_file_path = os.path.join(images_data_path, img_id.zfill(12) + '.txt')

        if not check_if_file_exist(path=txt_file_path):
            file_mode = 'w'
        else:
            file_mode = 'a'

        # Make a list of value to be written to txt file
        values_to_write = [category, bbox[0], bbox[1], bbox[2], bbox[3]]
        # Round to 6 decimal places (as in Yolo paper I saw)
        values_to_write = [round(item, 6) for item in values_to_write]
        #     values_to_write = tuple(values_to_write)
        with open(txt_file_path, file_mode) as f:
            f.write('{} {} {} {} {}\n'.format(*values_to_write))


def parse_and_save_to_yolo(json_abs_path, mode, images_data_path, extract_paperdoll_ids=False, root_data_dir_path=None):
    # Function to fetch, parse and save ModaNet data
    # Allows to extract images' ids to retrieve the images from
    # paperdoll dataset.
    # --------------------------------------
    # json_abs_path : absolute path to json that needs to be read
    #                 including file name
    # mode          : train or val
    #                 val does not contain any annotation
    # extract_paperdoll_ids : if to extract images ids for paperdoll dataset

    json_data = load_json(json_abs_path=json_abs_path)
    # print(json_abs_path)

    #     json_info = json_data['info']
    #     images_categories = json_data['categories']

    images_data = json_data['images']

    # TODO This data extraction for paperdoll dataset is either unnecessary or it is in completely in a wrong place here
    if extract_paperdoll_ids:
        images_to_retrieve = get_used_images_ids(identifier='id', data=images_data)
        np.save(os.path.join(root_data_dir_path, 'images_from_paperdoll_{}.npy'.format(mode)), images_to_retrieve)

    if mode == 'train':
        transform_annotations_to_yolo(json_data=json_data, images_data=images_data, images_data_path=images_data_path)

def main(args):
    ### PARAMETERS
    root_data_dir_path = args.root_data_dir_path
    # modanet_val = os.path.join(root_data_dir_path, 'modanet2018_instances_val.json')
    coco_json_filepath = os.path.join(root_data_dir_path, args.coco_json_filename)
    images_data_path = os.path.join(root_data_dir_path, args.images_dir_name)

    ### RUN
    parse_and_save_to_yolo(json_abs_path=coco_json_filepath, mode='train', images_data_path=images_data_path, extract_paperdoll_ids=False, root_data_dir_path=None)

    ## TO PLOT
    json_data = load_json(json_abs_path=coco_json_filepath)
    images_data = json_data['images']

    images_to_retrieve = get_used_images_ids(identifier='id', data=images_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir_path', type=str, required=True, default='/home/mwieczorek/visual_search/yolo/data/modanet/')
    parser.add_argument('--coco_json_filename', type=str, required=True, default='modanet2018_instances_train.json')
    parser.add_argument('--images_dir_name', type=str, required=True, default='train/')
    #parser.add_argument('--extract_paperdoll_ids', action="store_true", required=False, help='If true images (based on their id in JSON file) will be extracted to ')

    args = parser.parse_args()
    main(args)
