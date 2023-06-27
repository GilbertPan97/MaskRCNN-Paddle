import os
import paddle as pd
import paddlex as pdx
from paddlex import transforms as T


############################################################
#  Configurations
############################################################

# device used to train
DIVACE = 'gpu:0'    # gpu or cpu

# project root dir and src root dir
ROOT_DIR = os.path.abspath("./")

# coco dataset root dir(include annotations and images)
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "coco_dataset")

# output model dir(include epoch models and frozen model)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# train with coco weight or not
TRAIN_WITH_COCO = True

# if not train with coco weight, this need to be defined
# BASE_WEIGHT_PATH -> path/to/last/trained/model/weight
BASE_WEIGHT_PATH = []


# train configuration
class TrainConfig():
    # Train epochs
    EPOCHS = 36

    # Batch_size
    BatchSize = 2

    # Number of workers
    # NUM_WORKERS = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    NUM_WORKERS = 1

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on PaddlePaddle it causes
    # weights to explode. Likely due to differences in optimizer implementation.
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001


############################################################
# main function
############################################################
def main():
    # get train config
    config = TrainConfig()

    # specify the device for deep learning training
    pd.device.set_device(DIVACE)

    # Define transforms for training and validation
    # API docs：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.RandomResizeByShort(
            short_sizes=[640, 672, 704, 736, 768, 800],
            max_size=1333,
            interp='CUBIC'), T.RandomHorizontalFlip(), T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transforms = T.Compose([
        T.ResizeByShort(
            short_size=800, max_size=1333, interp='CUBIC'), T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the datasets used for training and validation
    # API docs: https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.CocoDetection(
        data_dir=DATA_ROOT_DIR + '/images',
        ann_file=DATA_ROOT_DIR + '/annotations/train.json',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.CocoDetection(
        data_dir=DATA_ROOT_DIR + '/images',
        ann_file=DATA_ROOT_DIR + '/annotations/val.json',
        transforms=eval_transforms)

    # Initialize the model and training.
    # Training indicators can be viewed using VisualDL, refer to:
    # https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/train/visualdl.md
    num_classes = len(train_dataset.labels)
    model = pdx.det.MaskRCNN(
        num_classes=num_classes, backbone='ResNet101', with_fpn=True)

    # API docs: https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/instance_segmentation.md
    # Parameter introduction and adjustment instructions:
    # https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/parameters.md
    model.train(
        num_epochs=config.EPOCHS,
        train_dataset=train_dataset,
        train_batch_size=config.BatchSize,
        eval_dataset=eval_dataset,
        pretrain_weights='COCO' if TRAIN_WITH_COCO else '',
        learning_rate=config.LEARNING_RATE,
        lr_decay_epochs=[24, 32],
        warmup_steps=20,
        warmup_start_lr=0.0,
        save_dir=MODELS_DIR + '/epoch_models',
        use_vdl=True)


if __name__ == "__main__":
    main()

# To visualization the training process, Please run following code in current terminal:
# visualdl --logdir ./models/epoch_models/vdl_log --port 8001

# To frozen model, Please run following code in current terminal:
# paddlex --export_inference --model_dir ./models/epoch_models/best_model --save_dir ./models

# To transfer labelme annotation to MSCOCO format, Please run following code in current terminal:
# paddlex --data_conversion --source labelme --to MSCOCO --pics path/to/your/images \
# --annotations path/to/your/annotation \
# --save_dir path/to/coco/save/dir

# To partition datasets, Please run following code in current terminal:
# paddlex --split_dataset --format COCO --dataset_dir path/to/coco/saved/dir \
# --val_value 0.2 --test_value 0.2


