import json, os, sys
from collections import defaultdict

# 用法: python coco2yolo_dogcat.py /home/xue/datasets/coco2017
root = sys.argv[1].rstrip("/")
img_dirs = {
    "train2017": f"{root}/images/train2017",
    "val2017": f"{root}/images/val2017"
}
ann_files = {
    "train2017": f"{root}/annotations/instances_train2017.json",
    "val2017": f"{root}/annotations/instances_val2017.json"
}

# 只保留 COCO 的 cat=15, dog=16，并映射到 0,1
keep_map = {15: 0, 16: 1}

def coco2yolo(split):
    out_dir = f"{root}/labels/{split}"
    os.makedirs(out_dir, exist_ok=True)
    with open(ann_files[split], "r") as f:
        coco = json.load(f)

    # id -> img meta
    imgs = {im["id"]: im for im in coco["images"]}
    # cat id -> 0/1
    cat_map = {c["id"]: keep_map[c["id"]] for c in coco["categories"] if c["id"] in keep_map}
    # img_id -> list[anns]
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1: 
            continue
        if a["category_id"] not in cat_map:
            continue
        anns_by_img[a["image_id"]].append(a)

    kept, empty = 0, 0
    for img_id, im in imgs.items():
        w, h = im["width"], im["height"]
        # 只处理存在于 images 目录、且有我们关心类别的图片
        anns = [a for a in anns_by_img.get(img_id, []) if a["category_id"] in keep_map]
        if not anns:
            # 没猫狗就不写标签文件，训练时会当背景图
            empty += 1
            continue
        # 写 YOLO 标签
        p = os.path.join(out_dir, os.path.splitext(im["file_name"])[0] + ".txt")
        with open(p, "w") as f:
            for a in anns:
                x, y, bw, bh = a["bbox"]
                # COCO bbox 是 x,y,w,h（左上角）；YOLO 需要 cx,cy,w,h，且归一化
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                f.write(f"{cat_map[a['category_id']]} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}\n")
        kept += 1
    print(f"[{split}] wrote labels: {kept}, skipped(no cat/dog): {empty}")

for sp in ["train2017", "val2017"]:
    coco2yolo(sp)
