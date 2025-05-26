from src.utils import load_data

IMG_DIR   = "data/images/xbox_controller"
LABEL_DIR = "data/labels/xbox_controller"

gen = load_data(IMG_DIR, LABEL_DIR)
img, boxes, cls = next(gen)
print("img shape:", img.shape)
print("boxes shape:", boxes.shape)
print("class IDs:", cls)