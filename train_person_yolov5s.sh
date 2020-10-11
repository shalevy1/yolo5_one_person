# python train.py --img 640 --batch 48 --epochs 100 --data ./data/person_coco.yaml --cfg ./models/yolov5s.yaml \
# --weights yolov5s.pt --single-cls --logdir persons/yolov5s --device=0 

python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 192 \
--epochs 100 --data ./data/person_coco.yaml --cfg ./models/yolov5s.yaml \
--weights yolov5s.pt --single-cls --logdir persons/yolov5s --device=0,1,2,3