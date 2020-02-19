import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from pycocotools import mask as pycocomask
from PIL import Image
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from predictor import KittiDemo

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

# run KITTI trk sequence 00-20
dataset_root = "/mnt/disk2/kitti-dataset/tracking"
split = "training"
camera_folder = "image_02"
seq_path = os.path.join(dataset_root, split, camera_folder)
seq_list = sorted(os.listdir(seq_path))
result_root = "/mnt/ssd2/maskrcnn/results/kitti/tracking"
result_folder = os.path.join(result_root, split)

for seq_id in tqdm(range(len(seq_list))):
    seq = seq_list[seq_id]
    seq_dir = os.path.join(seq_path, seq)
    img_files = sorted(os.listdir(seq_dir))
    result_dir = os.path.join(result_folder, seq)

    if not os.path.exists(os.path.join(result_dir, 'mrcnn_viz')):
        os.makedirs(os.path.join(result_dir, 'mrcnn_viz'))
    if not os.path.exists(os.path.join(result_dir, 'mrcnn_masks')):
        os.makedirs(os.path.join(result_dir, 'mrcnn_masks'))

    boxes_all = []
    scores_all = []
    labels_all = []
    masks_all = []
    img_idx_all = []

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
        masks = top_predictions.get_field("mask").numpy()

        for dt_id, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            boxes_all.append(box)
            labels_all.append(label)
            scores_all.append(score)
            mask = np.asfortranarray(mask[0].astype(np.uint8))
            mask_encode = pycocomask.encode(mask)
            mask = {'dt_id': dt_id, 'mask_encode': mask_encode}
            masks_all.append(mask)
            img_idx_all.append(idx)

        pred_img = image.copy()

        pred_img = kitti_demo.overlay_boxes(pred_img, top_predictions)
        if kitti_demo.cfg.MODEL.MASK_ON:
            pred_img = kitti_demo.overlay_mask(pred_img, top_predictions)
        result = kitti_demo.overlay_class_names(pred_img, top_predictions)

        cv2.imwrite('{}/{}/{}'.format(result_dir, 'mrcnn_viz', img_files[idx]), pred_img)

    with open(os.path.join(result_dir, 'mrcnn_dts.txt'), 'w') as f:
        for (box, score, label, mask, idx) in zip(boxes_all, scores_all, labels_all, masks_all, img_idx_all):
            f.write('{} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.4f}\n'.format(int(img_files[idx][:-4]), label, box[0], box[1],
                                                                        box[2], box[3], score))
            dt_id = mask['dt_id']
            mask_encode = mask['mask_encode']
            mask_path_encode = os.path.join(result_dir, 'mrcnn_masks',
                                            img_files[idx][:-4] + '_' + "{:02d}".format(dt_id) + '.pkl')
            with open(mask_path_encode, 'wb') as f_mask:
                pickle.dump(mask_encode, f_mask)
