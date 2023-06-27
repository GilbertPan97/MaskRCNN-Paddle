import os
import paddlex as pdx
import natsort

# project root dir and src root dir
ROOT_DIR = os.path.abspath('./')

# original images dir
ORI_IMGS_DIR = os.path.join(ROOT_DIR, 'images')

# frozen model dir
MODEL_DIR = os.path.join(ROOT_DIR, 'models/inference_model')

# inference result save dir
INFERENCE_DIR = os.path.join(ROOT_DIR, 'inference_results')

# inference thresh score
TH_SCORE = 0.5


if __name__ == '__main__':
    # get dir images name list
    imgs_list = os.listdir(ORI_IMGS_DIR)
    imgs_list = natsort.natsorted(imgs_list)

    for img_name in imgs_list:
        # get image path
        img_path = os.path.join(ORI_IMGS_DIR, img_name)

        model = pdx.load_model(MODEL_DIR)
        try:
            result = model.predict(img_path)
        except:
            continue
        else:
            print("Predict Result: ", result)
            pdx.det.visualize(img_path, result, threshold=TH_SCORE, save_dir=INFERENCE_DIR)