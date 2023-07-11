import os
import cv2
import paddle
import paddlex as pdx
from pycocotools import mask as maskUtils
# from pycocotools import COCO
import numpy as np
import natsort
import json

# images dir
ORI_IMGS_DIR = '../test_deli/images'

# model dir
MODEL_DIR = '../test_deli/best_model'

# json save path
COCO_PATH = '../test_deli/annotations/instances_default.json'


def binary_mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        polygon =[]
        for point in approx.squeeze():
            polygon.extend(point.tolist())

        polygons.append(polygon)

    return polygons


# def test_draw(image_path, coco_annotation_to_show):
#     img_mat = cv2.imread(image_path)
#     # coordinates
#     bbox = coco_annotation_to_show['bbox']
#     seg = coco_annotation_to_show['segmentation']
#     seg_coordinate = np.array([[seg[i], seg[i+1]] for i in range(np.floor(len(seg) / 2))], np.uint8)
#     seg_coordinate = seg_coordinate.reshape((-1, 1, 2))
#     # bounding box
#     # cv2.rectangle(img, )
#     # polygon
#     cv2.polylines(img_mat, [seg_coordinate], True, (0, 255, 255))
#
#     if True:
#         cv2.imshow("Box", img_mat)

def mask_to_coco_labels(mask, image_id, image_name, LabelCounter):
    masks = maskUtils.decode(mask)
    binary_masks = []

    for i in range(1):
        mask_i = masks[:, :]
        binary_mask = np.zeros_like(mask_i, dtype=np.uint8)
        binary_mask[mask_i > 0] = 1
        binary_masks.append(binary_mask)

    coco_annotations = []
    for i, bin_mask in enumerate(binary_masks):
        segmentation = binary_mask_to_polygon(bin_mask)
        if len(segmentation[0]) < 6 or len(segmentation) > 1:
            break
        # print("seg type: " + str(type(segmentation)))

        bin_mask_encode = maskUtils.encode(np.asfortranarray(bin_mask))
        LabelCounter.update()
        coco_annotation = {
            'id': LabelCounter.read(),            # annotation id
            'image_id': image_id,   # image id
            'category_id': 1,       # category id
            'segmentation': segmentation,
            'area': int(maskUtils.area(bin_mask_encode)),
            'bbox': maskUtils.toBbox(bin_mask_encode).tolist(),
            "iscrowd": 0,
            "attributes": {"occluded": False}
        }
        coco_annotations = coco_annotation

    return coco_annotations

class LabelCounter:
    ct = 0
    def update(self):
        self.ct = self.ct + 1

    def read(self):
        return self.ct

    def clear(self):
        self.ct = 0


if __name__ == '__main__':
    # get dir images name list
    imgs_list = os.listdir(ORI_IMGS_DIR)
    imgs_list = natsort.natsorted(imgs_list)
    coco_labels = {}
    coco_annotations = []

    # test only
    n = 0

    img_label_list = []
    lbcntr = LabelCounter()
    model = pdx.load_model(MODEL_DIR)

    for img_name in imgs_list:
        n = n + 1
        # if n > 594:
        #     break

        # get image path
        img_path = os.path.join(ORI_IMGS_DIR, img_name)
        img_mat = cv2.imread(img_path)

        # image label
        img_label = {
            'id': n,
            'width': img_mat.shape[1],
            'height': img_mat.shape[0],
            'file_name': img_name,
            'license': 0,
            'flickr_url': "",
            'coco_url': "",
            'date_captured': 0
        }
        img_label_list.append(img_label)

        print("Model inference on image: " + str(n) + ", filename: " + img_name)
        results = model.predict(img_path)
        if not results:
            print("No result detected on image: " + str(n) + ", filename: " + img_name)
            continue

        if n == 208 or n == 453:
            continue

        for result in results:
            mask_result = result['mask']

            coco_annotation = mask_to_coco_labels(mask_result, n, None, lbcntr)

            # # coordinates
            # # print(coco_annotation)
            # seg = coco_annotation[0]['segmentation'][0]
            # seg_coordinate = np.empty((int(len(seg) / 2), 2))
            # for i in range(int(len(seg) / 2)):
            #     seg_coordinate[i, :] = np.array([seg[2*i], seg[2*i+1]])
            # # seg_coordinate = np.array([[seg[i], seg[i + 1]] for i in range(int(len(seg) / 2))], np.uint8)
            # # print(seg_coordinate)
            # seg_coordinate = seg_coordinate.reshape((-1, 1, 2))
            # # print(seg_coordinate)
            #
            # # polygon
            # cv2.polylines(img_mat, np.int32([seg_coordinate]), True, (0, 255, 255), 2)
            #
            # if True:
            #     cv2.imshow("Box", img_mat)
            #     cv2.waitKey(500)
            if coco_annotation:
                coco_annotations.append(coco_annotation)

    coco_labels = {
        "licenses": [{
            "name": "",
            "id": 0,
            "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""},
        "categories": [{
            "id": 1,
            "name": "box",
            "supercategory": ""}],
        'images': img_label_list,
        'annotations': coco_annotations
    }
    coco_label_json = json.dumps(coco_labels)
    # print(coco_label_json)
    with open(COCO_PATH, "w") as output:
        output.write(coco_label_json)

    print("JSON file saved")
