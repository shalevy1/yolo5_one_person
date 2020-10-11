# python train.py --img 640 --batch 16 --epochs 5 --data ./data/person_coco.yaml --cfg ./models/yolov5l.yaml \
# --weights yolov5l.pt --single-cls --logdir persons/yolov5l --device=2,3

python -m torch.distributed.launch --nproc_per_node 3 train.py --batch-size 48 \
--epochs 100 --data ./data/person_coco.yaml --cfg ./models/yolov5l.yaml \
--weights yolov5l.pt --single-cls --logdir persons/yolov5l --device=1,2,3