from ultralytics import YOLO
import cv2
from image_processing import draw_box, resize_image
model = YOLO("/content/runs/segment/train/weights/best.pt")
results = model("/content/Surf_bag_segmentation-1/test/images/frame_422_jpg.rf.749184e03f6f90c898b5c28618b63398.jpg")
result = results[0]
class_list = model.model.names
labeled_img = draw_box(result.orig_img, result, class_list)
display_img = resize_image(labeled_img, 100)
cv2.imshow('image', display_img)