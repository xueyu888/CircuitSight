from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import typer
from PIL import Image, ImageOps

app = typer.Typer(add_completion=False)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _collect_images(images_dir: Path) -> List[Path]:
    files = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            files.append(p)
    if not files:
        raise typer.BadParameter(f"No images found in {images_dir}")
    return files

def _compute_starts(L: int, tile: int, stride: Optional[int], overlap: Optional[float]) -> List[int]:
    if stride is None:
        if overlap is None:
            overlap = 0.2
        stride = max(1, int(round(tile * (1 - overlap))))
    starts = list(range(0, max(1, L - tile + 1), stride))
    if not starts or starts[-1] != max(0, L - tile):
        starts.append(max(0, L - tile))
    return sorted(set(starts))

def _yolo_to_abs(xc, yc, w, h, W, H):
    return ((xc - w/2) * W, (yc - h/2) * H, (xc + w/2) * W, (yc + h/2) * H)

def _abs_to_yolo(x1, y1, x2, y2, W, H):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w/2.0
    yc = y1 + h/2.0
    return (xc / W, yc / H, w / W, h / H)

def _intersect(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _pad_to(img: Image.Image, tile_w: int, tile_h: int, mode: str) -> Image.Image:
    W, H = img.size
    if W == tile_w and H == tile_h:
        return img
    pad_w = tile_w - W
    pad_h = tile_h - H
    if mode == "edge":
        img = ImageOps.expand(img, border=(0,0,pad_w,pad_h), fill=None)
    else:
        img = ImageOps.expand(img, border=(0,0,pad_w,pad_h), fill=0)
    return img

def _load_labels(label_path: Path):
    if not label_path or not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0]); xc, yc, w, h = map(float, parts[1:])
        rows.append((cid, xc, yc, w, h))
    return rows

def _save_labels(path: Path, lines, save_empty: bool):
    if lines or save_empty:
        path.write_text("\n".join(lines))

@app.command()
def tile(
    images: Path = typer.Option(..., help="Input images dir"),
    out: Path = typer.Option(..., help="Output directory; will create images/ and labels/"),
    labels: Optional[Path] = typer.Option(None, help="Optional: YOLO labels dir aligned by filename"),
    tile_w: int = typer.Option(1024, help="Tile width"),
    tile_h: int = typer.Option(1024, help="Tile height"),
    overlap: Optional[float] = typer.Option(0.2, min=0.0, max=0.95, help="Overlap ratio (0~0.95). Ignored if stride specified."),
    stride_w: Optional[int] = typer.Option(None, help="Stride width in pixels"),
    stride_h: Optional[int] = typer.Option(None, help="Stride height in pixels"),
    min_area_frac: float = typer.Option(0.0002, help="Min intersection area fraction of tile to keep a label"),
    pad_mode: str = typer.Option("constant", help="Pad mode: constant|edge"),
    save_empty_labels: bool = typer.Option(True, help="Write empty labels when no objects"),
):
    images = images.expanduser().resolve()
    out = out.expanduser().resolve()
    if labels:
        labels = labels.expanduser().resolve()

    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    files = _collect_images(images)

    for img_path in files:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        sx_list = _compute_starts(W, tile_w, stride_w, overlap)
        sy_list = _compute_starts(H, tile_h, stride_h, overlap)

        yolo_path = (labels / (img_path.stem + ".txt")) if labels else None
        yolo_lbls = _load_labels(yolo_path)

        for y0 in sy_list:
            for x0 in sx_list:
                x1 = min(W, x0 + tile_w)
                y1 = min(H, y0 + tile_h)
                crop = img.crop((x0, y0, x1, y1))
                crop = _pad_to(crop, tile_w, tile_h, pad_mode)

                base = f"{img_path.stem}_x{x0:05d}_y{y0:05d}"
                out_img = out_images / f"{base}.png"
                out_lbl = out_labels / f"{base}.txt"
                crop.save(out_img)

                rows_out = []
                if yolo_lbls:
                    tile_box = (x0, y0, x0 + tile_w, y0 + tile_h)
                    tile_area = tile_w * tile_h
                    for cid, xc, yc, w, h in yolo_lbls:
                        ax1, ay1, ax2, ay2 = _yolo_to_abs(xc, yc, w, h, W, H)
                        inter = _intersect((ax1, ay1, ax2, ay2), tile_box)
                        if inter is None:
                            continue
                        ix1, iy1, ix2, iy2 = inter
                        # move into tile coords
                        tx1 = max(0.0, min(float(tile_w), ix1 - x0))
                        ty1 = max(0.0, min(float(tile_h), iy1 - y0))
                        tx2 = max(0.0, min(float(tile_w), ix2 - x0))
                        ty2 = max(0.0, min(float(tile_h), iy2 - y0))
                        inter_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
                        if inter_area / tile_area < min_area_frac:
                            continue
                        nxc, nyc, nw, nh = _abs_to_yolo(tx1, ty1, tx2, ty2, tile_w, tile_h)
                        if nw > 0 and nh > 0:
                            rows_out.append(f"{cid} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")

                _save_labels(out_lbl, rows_out, save_empty_labels)

    typer.echo(f"Done. Output at: {out}")

if __name__ == "__main__":
    app()
