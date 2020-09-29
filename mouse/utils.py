import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import deeplabcut
import json
import skimage.io
from skimage.util import img_as_ubyte, img_as_float
from skimage import morphology, measure, filters
from shutil import copyfile
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.morphology import square, dilation
from skimage.color import rgb2gray
from .mouse import MouseDataset
from .mouse import InferenceConfig
from .shape import shapes_to_labels_masks

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib


def video2frames(video_dir):
    """Convert a video into frames saved in a directory named as the video name.
    video_dir: path to the video
    """
    cap = cv2.VideoCapture(video_dir)
    nframes = int(cap.get(7))

    data_dir = os.path.splitext(video_dir)[0]
    frames_dir = os.path.join(data_dir, "images")

    os.mkdir(data_dir)
    os.mkdir(frames_dir)

    for index in range(nframes):
        cap.set(1, index)  # extract a particular frame
        ret, frame = cap.read()
        if ret:
            image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            img_name = os.path.join(frames_dir, str(index) + ".jpg")

            skimage.io.imsave(img_name, image)


def background_subtraction(frames_dir, background_dir):
    """Generate foregrounds corresponding to frames
    frames_dir: path to directory containing frames
    background_dir: path to the background image
    return:
        components: 1D array of number of blobs in each frame.
    """
    fg_dir = os.path.join(os.path.dirname(frames_dir), 'FG')
    os.mkdir(fg_dir)

    bg = img_as_float(skimage.io.imread(background_dir))
    if bg.ndim == 3:
        bg = rgb2gray(bg)

    threshold = bg * 0.5

    frames_list = os.listdir(frames_dir)
    components = np.zeros(len(frames_list), dtype=int)

    for frame in range(len(frames_list)):
        im = img_as_float(skimage.io.imread(os.path.join(frames_dir, str(frame) + '.jpg')))

        if im.ndim == 3:
            im = rgb2gray(im)

        fg = (bg - im) > threshold
        bw1 = morphology.remove_small_objects(fg, 1000)
        bw2 = morphology.binary_closing(bw1, morphology.disk(radius=10))
        bw3 = morphology.binary_opening(bw2, morphology.disk(radius=10))
        label = measure.label(bw3)
        num_fg = np.max(label)

        masks = np.zeros([bg.shape[0], bg.shape[1], 3], dtype=np.uint8)

        if num_fg == 2:
            bw3_1 = label == 1
            bw4_1 = morphology.binary_closing(bw3_1, morphology.disk(radius=30))
            bw5_1 = filters.median(bw4_1, morphology.disk(10))

            bw3_2 = label == 2
            bw4_2 = morphology.binary_closing(bw3_2, morphology.disk(radius=30))
            bw5_2 = filters.median(bw4_2, morphology.disk(10))

            # masks[:, :, 0] = img_as_bool(bw5_1)
            # masks[:, :, 1] = img_as_bool(bw5_2)
            masks[:, :, 0] = img_as_ubyte(bw5_1)
            masks[:, :, 1] = img_as_ubyte(bw5_2)
        else:
            masks[:, :, 0] = img_as_ubyte(bw3)

        components[frame] = num_fg
        # masks = masks.astype(np.uint8)
        skimage.io.imsave(os.path.join(fg_dir, str(frame) + '.png'), masks)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(frames_dir), 'components.csv'), index=False)

    return components


def split_train_val(dataset_dir, frac_split_train):
    """Split a dataset into subsets train and val inside dataset directory
    dataset_dir: path to the dataset containing images and their annotation json files
    frac_split_train: fraction of train subset in the dataset
    """
    json_ids = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    random.shuffle(json_ids)

    train_dir = os.path.join(dataset_dir, 'train')
    os.mkdir(train_dir)

    val_dir = os.path.join(dataset_dir, 'val')
    os.mkdir(val_dir)

    for json_id in json_ids[: int(frac_split_train * len(json_ids))]:
        copyfile(os.path.join(dataset_dir, json_id), os.path.join(train_dir, json_id))
        os.remove(os.path.join(dataset_dir, json_id))

        copyfile(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'),
                 os.path.join(train_dir, os.path.splitext(json_id)[0] + '.jpg'))
        os.remove(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'))

    for json_id in json_ids[int(frac_split_train * len(json_ids)):]:
        copyfile(os.path.join(dataset_dir, json_id), os.path.join(val_dir, json_id))
        os.remove(os.path.join(dataset_dir, json_id))

        copyfile(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'),
                 os.path.join(val_dir, os.path.splitext(json_id)[0] + '.jpg'))
        os.remove(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'))


def create_dataset(images_dir, components_info, num_annotations):
    """Randomly choose images which have one blob in their foreground
    images_dir: path to images directory
    components_info: path to a csv file or an array
    num_annotations: the number of images will be picked
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    dataset_dir = os.path.join(os.path.dirname(images_dir), 'dataset')
    os.mkdir(dataset_dir)

    touching = [i for i in range(len(components)) if components[i] == 1]
    if (components == 1).sum() > num_annotations:
        random.shuffle(touching)
        for image_id in touching[:num_annotations]:
            copyfile(os.path.join(images_dir, str(image_id) + '.jpg'),
                     os.path.join(dataset_dir, str(image_id) + '.jpg'))
    else:
        for image_id in touching:
            copyfile(os.path.join(images_dir, str(image_id) + '.jpg'),
                     os.path.join(dataset_dir, str(image_id) + '.jpg'))


def correct_segmentation_errors(components_info, fix_dir, frames_dir):
    """Count and pick one failed frame in every 3 consecutive fail frames for correcting"
    components_info: path to a csv file or an array
    fix_dir: path to directory for saving frames chosen
    frames_dir: path to directory containing frames
    return:
        correct_frames: the number of frames picked up
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    errors = np.array(components != 2, dtype=int)
    errors_accumulate = np.zeros(len(errors))
    interval_start = 0

    for i in range(len(errors)):
        if (errors[i] == 1) & (interval_start == 0):
            interval_start = 1
        elif errors[i] == 0:
            interval_start = 0

        if (interval_start == 1) & (i > 0):
            errors_accumulate[i] = errors_accumulate[i - 1] + 1

    # plt.plot(errors_accumulate)
    correct_frames = 0

    if components[0] != 2:
        copyfile(os.path.join(frames_dir, '0.jpg'), os.path.join(fix_dir, '0.jpg'))
        correct_frames = correct_frames + 1

    for i in range(len(errors_accumulate)):
        if (errors_accumulate[i] > 0) & (errors_accumulate[i] % 3 == 0):
            copyfile(os.path.join(frames_dir, str(i) + '.jpg'), os.path.join(fix_dir, str(i) + '.jpg'))
            correct_frames = correct_frames + 1
    return correct_frames


def tracking_inference(fg_dir, components_info):
    """Track the identities of mice
    fg_dir: path to directory containing foreground
    components_info: path to a csv file or an array
    """
    tracking_dir = os.path.join(os.path.dirname(fg_dir), 'tracking')
    os.mkdir(tracking_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    I = skimage.io.imread(os.path.join(fg_dir, str(0) + '.png'))
    skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)

    I = img_as_ubyte(I/255)

    for i in range(1, components.shape[0]):
        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        if components[i] == 2:
            J = skimage.io.imread(os.path.join(fg_dir, str(i) + '.png')) / 255.0

            J1 = J[:, :, 0]
            J2 = J[:, :, 1]

            overlap_1 = np.sum(np.multiply(J1, I1)[:]) / np.sum(I1[:])
            overlap_2 = np.sum(np.multiply(J2, I1)[:]) / np.sum(I1[:])
            overlap_12 = np.abs(overlap_1 - overlap_2)

            overlap_3 = np.sum(np.multiply(J1, I2)[:]) / np.sum(I2[:])
            overlap_4 = np.sum(np.multiply(J2, I2)[:]) / np.sum(I2[:])
            overlap_34 = np.abs(overlap_3 - overlap_4)

            if overlap_12 >= overlap_34:
                if overlap_1 >= overlap_2:
                    I[:, :, 0] = J1
                    I[:, :, 1] = J2
                else:
                    I[:, :, 0] = J2
                    I[:, :, 1] = J1
            else:
                if overlap_3 >= overlap_4:
                    I[:, :, 1] = J1
                    I[:, :, 0] = J2
                else:
                    I[:, :, 1] = J2
                    I[:, :, 0] = J1

            I = I.astype(np.uint8) * 255
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)

        else:
            I = I.astype(np.unit8) * 255
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)


def mask_based_detection(tracking_dir, components_info, floor=[[71, 71], [470, 470]], image_shape=(540, 540)):
    """Detect snout and tailbase coordinated from masks
    tracking_dir: path to directory containing masks corresponding to identities
    components_info: path to a csv file or an array
    floor: coordinates of top left and bottom right corners of rectangular floor zone
    image_shape: size of frames (height, width)
    return:
        np.array(features_mouse1_df): coordinates of snout and tailbase of mouse 1
        np.array(features_mouse2_df): coordinates of snout and tailbase of mouse 2
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    features_mouse1 = np.zeros((len(components), 4))
    features_mouse2 = np.zeros((len(components), 4))

    floor_zone = np.zeros(image_shape)
    floor_zone[floor[0][0]:floor[1][0], floor[0][1]:floor[1][1]] = 1

    for i in range(len(components)):
        I = (skimage.io.imread(os.path.join(tracking_dir, str(i) + '.png')) / 255.0).astype(int)

        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        properties1 = regionprops(I1.astype(int), I1.astype(float))
        center_of_mass1 = properties1[0].centroid

        properties2 = regionprops(I2.astype(int), I2.astype(float))
        center_of_mass2 = properties2[0].centroid

        BB1 = find_contours(I1, 0.5)[0]
        BB2 = find_contours(I2, 0.5)[0]

        # mouse 1
        center_BB1 = np.sum((BB1 - center_of_mass1) ** 2, axis=1)
        index1 = np.argmax(center_BB1)
        I1_end1 = BB1[index1]

        end1_BB1 = np.sum((BB1 - I1_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB1)
        I1_end_max = np.max(end1_BB1)
        I1_end2 = BB1[index2]

        condition_mouse1 = np.sum(np.multiply(floor_zone, I1)[:]) / np.sum(I1[:])

        if i == 0:
            features_mouse1[i, :2] = I1_end1
            features_mouse1[i, 2:] = I1_end2
        else:
            if ((I1_end_max >= 90) & (condition_mouse1 == 1)):
                features_mouse1[i, :2] = I1_end1
                features_mouse1[i, 2:] = I1_end2
            else:
                end1_nose = np.sum((I1_end1 - features_mouse1[i - 1, :2]) ** 2)
                end1_tail = np.sum((I1_end1 - features_mouse1[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse1[i, :2] = I1_end1
                    features_mouse1[i, 2:] = I1_end2
                else:
                    features_mouse1[i, :2] = I1_end2
                    features_mouse1[i, 2:] = I1_end1

                    # mouse 2
        center_BB2 = np.sum((BB2 - center_of_mass2) ** 2, axis=1)
        index1 = np.argmax(center_BB2)
        I2_end1 = BB2[index1]

        end1_BB2 = np.sum((BB2 - I2_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB2)
        I2_end_max = np.max(end1_BB2)
        I2_end2 = BB2[index2]

        condition_mouse2 = np.sum(np.multiply(floor_zone, I2)[:]) / np.sum(I2[:])

        if i == 0:
            features_mouse2[i, :2] = I2_end1
            features_mouse2[i, 2:] = I2_end2
        else:
            if ((I2_end_max >= 90) & (condition_mouse2 == 1)):
                features_mouse2[i, :2] = I2_end1
                features_mouse2[i, 2:] = I2_end2
            else:
                end1_nose = np.sum((I2_end1 - features_mouse2[i - 1, :2]) ** 2)
                end1_tail = np.sum((I2_end1 - features_mouse2[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse2[i, :2] = I2_end1
                    features_mouse2[i, 2:] = I2_end2
                else:
                    features_mouse2[i, :2] = I2_end2
                    features_mouse2[i, 2:] = I2_end1

    features_mouse1 = np.round(features_mouse1, 2)

    features_mouse1_df = pd.DataFrame({'snout_x': features_mouse1[:, 1],
                                       'snout_y': features_mouse1[:, 0],
                                       'tailbase_x': features_mouse1[:, 3],
                                       'tailbase_y': features_mouse1[:, 2]})
    features_mouse1_df.to_csv(os.path.join(os.path.dirname(tracking_dir), 'features_mouse1_md.csv'),
                              index=False)

    features_mouse2 = np.round(features_mouse2, 2)
    features_mouse2_df = pd.DataFrame({'snout_x': features_mouse2[:, 1],
                                       'snout_y': features_mouse2[:, 0],
                                       'tailbase_x': features_mouse2[:, 3],
                                       'tailbase_y': features_mouse2[:, 2]})
    features_mouse2_df.to_csv(os.path.join(os.path.dirname(tracking_dir), 'features_mouse2_md.csv'),
                              index=False)

    return np.array(features_mouse1_df), np.array(features_mouse2_df)


def mice_separation(tracking_dir, frames_dir, bg_dir):
    """Separate the sequence of frames into 2 videos. Each video contains one mouse
    tracking_dir: path to directory containing masks corresponding to identities
    frames_dir: path to frames directory
    bg_dir: path to the background image
    """
    bg = img_as_ubyte(skimage.io.imread(bg_dir))
    num_images = len(os.listdir(tracking_dir))

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    video_mouse1_dir = os.path.join(os.path.dirname(tracking_dir), 'mouse1.avi')
    video1 = cv2.VideoWriter(video_mouse1_dir, fourcc, 30, (bg.shape[1], bg.shape[0]), 0)

    video_mouse2_dir = os.path.join(os.path.dirname(tracking_dir), 'mouse2.avi')
    video2 = cv2.VideoWriter(video_mouse2_dir, fourcc, 30, (bg.shape[1], bg.shape[0]), 0)

    for i in range(num_images):
        masks = skimage.io.imread(os.path.join(tracking_dir, str(i) + '.png')) / 255
        image = skimage.io.imread(os.path.join(frames_dir, str(i) + '.jpg'))

        mask1 = masks[:, :, 0].astype(np.uint8)
        mask1 = dilation(mask1, square(10))

        mask2 = masks[:, :, 1].astype(np.uint8)
        mask2 = dilation(mask2, square(10))

        mouse2_remove = (mask2 != 1) | (mask1 == 1)
        mouse1 = np.multiply(image, mouse2_remove) + np.multiply(bg, (1 - mouse2_remove))
        mouse1 = img_as_ubyte(mouse1)

        mouse1_remove = (mask1 != 1) | (mask2 == 1)
        mouse2 = np.multiply(image, mouse1_remove) + np.multiply(bg, (1 - mouse1_remove))
        mouse2 = img_as_ubyte(mouse2)

        video1.write(mouse1)
        video2.write(mouse2)

    cv2.destroyAllWindows()
    video1.release()
    video2.release()


def deeplabcut_detection(config_dir, video_dir):
    """Detect snout and tailbase coordinated with Deeplabcut model
    config_dir: path to config file
    video_dir: path to video input
    return:
        features_mouse1: coordinates of snout and tailbase of mouse 1
        features_mouse2: coordinates of snout and tailbase of mouse 2
    """
    deeplabcut.analyze_videos(config_dir, video_dir, videotype='.avi')

    dlc_output = [f for f in os.listdir(os.path.dirname(video_dir[0])) if f.endswith('.h5')]


    # mouse1
    mouse1_dlc = pd.read_hdf(os.path.join(os.path.dirname(video_dir[0]), dlc_output[0]))
    features_mouse1 = mouse1_dlc.values[:, [0, 1, 9, 10]]
    features_mouse1 = np.round(features_mouse1, 2)
    features_mouse1_df = pd.DataFrame({'snout_x': np.round(mouse1_dlc.values[:, 0], 2),
                                       'snout_y': np.round(mouse1_dlc.values[:, 1], 2),
                                       'tailbase_x': np.round(mouse1_dlc.values[:, 9], 2),
                                       'tailbase_y': np.round(mouse1_dlc.values[:, 10], 2)})
    features_mouse1_df.to_csv(os.path.join(os.path.dirname(video_dir[0]), 'features_mouse1_dlc.csv'),
                              index=False)

    mouse2_dlc = pd.read_hdf(os.path.join(os.path.dirname(video_dir[0]), dlc_output[1]))
    features_mouse2 = mouse2_dlc.values[:, [0, 1, 9, 10]]
    features_mouse2 = np.round(features_mouse2, 2)
    features_mouse2_df = pd.DataFrame({'snout_x': np.round(mouse2_dlc.values[:, 0], 2),
                                       'snout_y': np.round(mouse2_dlc.values[:, 1], 2),
                                       'tailbase_x': np.round(mouse2_dlc.values[:, 9], 2),
                                       'tailbase_y': np.round(mouse2_dlc.values[:, 10], 2)})
    features_mouse2_df.to_csv(os.path.join(os.path.dirname(video_dir[0]), 'features_mouse2_dlc.csv'),
                              index=False)

    return features_mouse1, features_mouse2


def ensemble_features(features_mouse_md, features_mouse_dlc, tracking_dir, mouse_id=1):
    """Ensemble the result of mask-based detection and deeplabcut-based detection
    features_mouse_md: coordinates of snout and tailbase generated by mask-based detection
    features_mouse_dlc: coordinates of snout and tailbase generated by deeplabcut detection
    tracking_dir: path to directory containing masks corresponding to identities
    mouse_id: mouse id ( 1 or 2)
    return:
        features_ensemble: ensemble coordinates of snout and tailbase
    """
    features_ensemble = np.zeros(features_mouse_md.shape)
    for i in range(len(features_mouse_md)):
        masks = skimage.io.imread(os.path.join(tracking_dir, str(i) + '.png')) / 255.0

        mask = masks[:, :, mouse_id - 1].astype(int)
        mask = dilation(mask, square(15))

        nose_DLC = np.zeros(mask.shape)
        tailbase_DLC = np.zeros(mask.shape)

        nose_DLC[int(features_mouse_dlc[i, 1]), int(features_mouse_dlc[i, 0])] = 1
        tailbase_DLC[int(features_mouse_dlc[i, 3]), int(features_mouse_dlc[i, 2])] = 1

        if np.sum(np.multiply(mask, nose_DLC)[:]) > 0:
            features_ensemble[i, :2] = features_mouse_dlc[i, :2]
        else:
            features_ensemble[i, :2] = features_mouse_md[i, :2]

        if np.sum(np.multiply(mask, tailbase_DLC)[:]) > 0:
            features_ensemble[i, 2:] = features_mouse_dlc[i, 2:]
        else:
            features_ensemble[i, 2:] = features_mouse_md[i, 2:]

    features_ensemble_df = pd.DataFrame({'snout_x': features_ensemble[:, 0],
                                         'snout_y': features_ensemble[:, 1],
                                         'tailbase_x': features_ensemble[:, 2],
                                         'tailbase_y': features_ensemble[:, 3]})
    features_ensemble_df.to_csv(os.path.join(os.path.dirname(tracking_dir),
                                             'features_mouse' + str(mouse_id) + '_ensemble.csv'), index=False)

    return features_ensemble


def labelmejson_to_png(fix_dir, output_dir):
    """Convert annotations created by labelme to images
    fix_dir: path to directory for saving frames chosen
    output_dir: path to save output
    """
    json_ids = [f for f in os.listdir(fix_dir) if f.endswith('.json')]

    dataset_fix = MouseDataset()
    dataset_fix.load_mouse(fix_dir, "")

    class_name_to_id = {label["name"]: label["id"] for label in dataset_fix.class_info}
    # Read mask file from json
    for json_id in json_ids:
        json_id_dir = os.path.join(fix_dir, json_id)
        with open(json_id_dir) as f:
            data = json.load(f)
            image_shape = (data['imageHeight'], data['imageWidth'])

            cls, masks = shapes_to_labels_masks(img_shape=image_shape,
                                                shapes=data['shapes'],
                                                label_name_to_value=class_name_to_id)

        masks_rgb = np.zeros((data['imageHeight'], data['imageWidth'], 3), dtype=np.float)
        masks_rgb[:, :, :2] = masks[:, :, :2]
        skimage.io.imsave(os.path.join(output_dir, os.path.splitext(json_id)[0] + '.png'), masks_rgb)


def mouse_mrcnn_segmentation(components_info, frames_dir, background_dir, model_dir, model_path=None):
    """Segment mice using Mask-RCNN model
    components_info: path to a csv file or an array
    frames_dir: path to frames directory
    background_dir: path to background image
    model_dir: path to save log and trained model
    model_path: path to model weights
    return:
        components: array of the number of blobs in each frames
    """
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

    if model_path:
        model.load_weights(model_path, by_name=True)
    else:
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)

    output_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bg = cv2.imread(background_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    print("The video has {} frames: ".format(components.shape[0]))

    for i in range(components.shape[0]):
        if components[i] != 2:
            image_name = str(i) + '.jpg'
            image = skimage.io.imread(frames_dir + '/' + image_name)

            if image.ndim == 2:
                image_rgb = skimage.color.gray2rgb(image)
            else:
                image_rgb = image

            results = model.detect([image_rgb], verbose=1)

            results_package = results[0]

            masks_rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)

            if len(results_package["scores"]) >= 2:
                class_ids = results_package['class_ids'][:2]
                scores = results_package['scores'][:2]
                masks = results_package['masks'][:, :, :2]  # Bool
                rois = results_package['rois'][:2, :]

                masks_1 = morphology.remove_small_objects(masks[:, :, 0], 1000)
                masks_2 = morphology.remove_small_objects(masks[:, :, 1], 1000)

                if (masks_1.sum().sum() > 0) & (masks_2.sum().sum() > 0):
                    masks_rgb[:, :, 0] = img_as_ubyte(masks_1)
                    masks_rgb[:, :, 1] = img_as_ubyte(masks_2)

                    components[i] = 2

            skimage.io.imsave(os.path.join(output_dir, str(i) + '.png'), masks_rgb)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(frames_dir), 'components.csv'), index=False)

    return components

def check_mrcnn_model_path(model_dir):
    dir_names = next(os.walk(model_dir))[1]
    key = "mask_rcnn"
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return False
    else:
        return True
