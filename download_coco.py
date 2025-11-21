import json, shutil, os

SRC_IMG = '/path/to/coco/images/train2017'
SRC_ANN = '/path/to/coco/annotations/instances_train2017.json'
DST_IMG = 'data/coco_catdog/images/train'
DST_LABEL = 'data/coco_catdog/labels/train'
os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LABEL, exist_ok=True)

TARGET_CLASSES = ['cat', 'dog']

# 1. 读类别信息
with open(SRC_ANN) as f:
    coco = json.load(f)
cat_id_map = {c['id']: c['name'] for c in coco['categories']}
target_ids = [k for k, v in cat_id_map.items() if v in TARGET_CLASSES]

# 2. 过滤标注
for ann in coco['annotations']:
    if ann['category_id'] not in target_ids:
        continue
    img_info = next(i for i in coco['images'] if i['id'] == ann['image_id'])
    img_path = os.path.join(SRC_IMG, img_info['file_name'])
    if not os.path.exists(img_path):
        continue
    dst_img_path = os.path.join(DST_IMG, img_info['file_name'])
    shutil.copy(img_path, dst_img_path)

    # 写 YOLO 格式标签
    w, h = img_info['width'], img_info['height']
    bbox = ann['bbox']  # [x,y,w,h]
    xc, yc = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
    with open(os.path.join(DST_LABEL, img_info['file_name'].replace('.jpg', '.txt')), 'a') as f:
        f.write(f"{target_ids.index(ann['category_id'])} {xc/w:.6f} {yc/h:.6f} {bbox[2]/w:.6f} {bbox[3]/h:.6f}\n")
