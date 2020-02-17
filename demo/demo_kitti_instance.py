from maskrcnn_benchmark.config import cfg
from predictor import KittiDemo
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


config_file = "../configs/kitti/e2e_mask_rcnn_R_50_FPN_1x_kitti_instance.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)

# mannual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
cfg.merge_from_list(["MODEL.WEIGHT",
                     "/mnt/nas_crdataset/models/mrcnn_model/kitti/model_0004000.pth"])

kitti_demo = KittiDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.9,
    show_mask_heatmaps=False,
)

category_id_to_contiguous_id = {
    'background': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'caravan': 4,
    'motorcycle': 5,
    'person': 6,
    'rider': 7,
    'trailer': 8,
    'train': 9,
    'truck': 10
}

# kitti_demo.CATEGORIES = list(category_id_to_contiguous_id.keys())

# load images and then run prediction
# img_dir = "/mnt/ht/kitti/semantics/data_semantics/training/image_2"
# img_files = sorted(os.listdir(img_dir))
#
# for idx in random.sample(range(1, len(img_files)), 10):
#     img = Image.open(os.path.join(img_dir, img_files[idx])).convert("RGB")
#     image = np.array(img)[:, :, [2, 1, 0]]
#     predictions = kitti_demo.run_on_opencv_image(image)
#
#     imshow(predictions)

# for idx in range(len(img_files)):
#     img = Image.open(os.path.join(img_dir, img_files[idx])).convert("RGB")
#     image = np.array(img)[:, :, [2, 1, 0]]
#     predictions = kitti_demo.run_on_opencv_image(image)
#
#     imshow(predictions)
# cv2.imwrite('/mnt/ht/ECCV/experiments/2019_09_29_onrd041/results/' + str(idx).zfill(10) + '.jpg', predictions)

# run KITTI trk sequence 00-20
seq_folder = '/mnt/disk2/kitti-dataset/raw_data/2011_09_26/2011_09_26_drive_0091_sync/image_02'
result_folder = '/mnt/ssd2/maskrcnn/results/kitti/raw_data/2011_09_26/2011_09_26_drive_0091_sync/image_02'
# seq_list = sorted(os.listdir(seq_folder))
seq_list = ['data']

for seq_id in tqdm(range(len(seq_list))):
    seq = seq_list[seq_id]
    seq_dir = os.path.join(seq_folder, seq)
    img_files = sorted(os.listdir(seq_dir))

    boxes_all = []
    scores_all = []
    labels_all = []

    for idx in range(len(img_files)):
        img = Image.open(os.path.join(seq_dir, img_files[idx])).convert("RGB")
        image = np.array(img)[:, :, [2, 1, 0]]
        img_shape = image.shape[1], image.shape[0]
        prediction = kitti_demo.compute_prediction(image)
        top_predictions = kitti_demo.select_top_predictions(prediction)
        top_predictions = kitti_demo.select_interest_labels(top_predictions)
        # top_predictions = kitti_demo.merge_rider_bicycle_labels(top_predictions)

        boxes = top_predictions.bbox.tolist()
        labels = top_predictions.get_field("labels").tolist()
        scores = top_predictions.get_field("scores").tolist()

        for (box, label, score) in zip(boxes, labels, scores):
            boxes_all.append(box)
            labels_all.append(label)
            scores_all.append(score)

        pred_img = image.copy()

        pred_img = kitti_demo.overlay_boxes(pred_img, top_predictions)
        if kitti_demo.cfg.MODEL.MASK_ON:
            pred_img = kitti_demo.overlay_mask(pred_img, top_predictions)
        result = kitti_demo.overlay_class_names(pred_img, top_predictions)

        # if not os.path.exists(os.path.join(result_folder, seq)):
        #     os.makedirs(os.path.join(result_folder, seq))
        # cv2.imwrite('{}/{}/{}'.format(result_folder, seq, img_files[idx]), pred_img)
        if not os.path.exists(os.path.join(result_folder)):
            os.makedirs(os.path.join(result_folder))
        cv2.imwrite('{}/{}'.format(result_folder, img_files[idx]), pred_img)

    # with open(os.path.join(result_folder, seq + '.txt'), 'w') as f:
    #     for (box, score, label, idx) in zip(boxes_all, scores_all, labels_all, range(len(img_files))):
    #         f.write('{},{},{},{},{},{},{} \n'.format(int(img_files[idx][:-4]), label, box[0], box[1], box[2], box[3],
    #                                               score))
    #     f.close()
    with open(os.path.join(result_folder + '.txt'), 'w') as f:
        for (box, score, label, idx) in zip(boxes_all, scores_all, labels_all, range(len(img_files))):
            f.write('{},{},{},{},{},{},{} \n'.format(int(img_files[idx][:-4]), label, box[0], box[1], box[2], box[3],
                                                  score))
        f.close()
