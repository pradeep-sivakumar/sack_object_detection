import os
from dotenv import load_dotenv

import cv2
from ultralytics import YOLO

from image_processing import draw_box, resize_image

class ObjectDetector:
    def __init__(self):
        self.model_path = os.getenv("YOLO_MODEL_PATH", None)  # Default path
        if self.model_path is None:
            raise ValueError("YOLO_MODEL_PATH environment variable is not set.")
        self.model = YOLO(self.model_path)
        self.class_list = self.model.model.names

    def detect_objects(self, image_path):
        results = self.model(image_path)
        result = results[0]
        labeled_img = draw_box(result.orig_img, result, self.class_list)
        # display_img = resize_image(labeled_img, 100)
        return labeled_img, result

    def display_image(self, image):
        cv2.imshow('outputimage', image)
        cv2.waitKey(0)  # Wait for user input to close the window

if __name__ == "__main__":
    detector = ObjectDetector()
    image_path = "/content/Surf_bag_segmentation-1/test/images/frame_422_jpg.rf.749184e03f6f90c898b5c28618b63398.jpg"  # Replace with your image path
    labeled_img, result = detector.detect_objects(image_path)
    detector.display_image(labeled_img)
    cv2.destroyAllWindows()  # Clean up windows
