import os
import paddlex as pdx
import natsort
from paddlex import transforms as T
from pycocotools.coco import COCO

# project root dir and src root dir
ROOT_DIR = os.path.abspath('./')

# frozen model dir
MODEL_DIR = os.path.join(ROOT_DIR, 'models/inference_model')

# coco dataset root dir(include annotations and images)
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "coco_dataset1")

# inference thresh score
TH_SCORE = 0.5

# inference with test dataset or not
INFER_WITH_TEST_DATASET = True
ORI_IMGS_DIR = os.path.join(ROOT_DIR, 'images')                 # if not inference with test dataset, set to image dir
INFERENCE_DIR = os.path.join(ROOT_DIR, 'inference_results')     # inference result save dir


if __name__ == '__main__':

    # get dir images name list
    if INFER_WITH_TEST_DATASET:

        coco = COCO(os.path.join(DATA_ROOT_DIR, 'annotations', 'test.json'))
        img_dict = coco.loadImgs(coco.getImgIds())
        imgs_folder = os.path.join(DATA_ROOT_DIR, 'images')
        imgs_list = []
        for img in img_dict:
            img_path = os.path.join(imgs_folder, img['file_name'])
            imgs_list.append(img_path)
    else:
        imgs_names = os.listdir(ORI_IMGS_DIR)
        imgs_names = natsort.natsorted(imgs_names)

        imgs_list = []
        for i in range(len(imgs_names)):
            imgs_list.append(os.path.join(ORI_IMGS_DIR, imgs_names[i]))

    # load inference model
    model = pdx.load_model(MODEL_DIR)

    # inference execution
    for img_path in imgs_list:

        try:
            result = model.predict(img_path)
        except:
            continue
        else:
            print("Predict Result: ", result)
            pdx.det.visualize(img_path, result, threshold=TH_SCORE, save_dir=INFERENCE_DIR)