import os
import numpy as np
from PIL import Image

PATH_TO_MODEL_PREDICTION = "./satellite-image-segmentation/mapbox_india_data/data_first_150/results"
PATH_TO_EXPECTED_PREDICTION = "./satellite-image-segmentation/mapbox_india_data/data_first_150/labels/test"
EVALUATE_METRICS = ['dice', 'jaccard', 'accuracy']


def main():
    predict_files = get_list_of_files(PATH_TO_MODEL_PREDICTION)
    predict_files.sort()

    expect_files = get_list_of_files(PATH_TO_EXPECTED_PREDICTION)
    expect_files.sort()

    if len(predict_files) != len(expect_files):
        print("Number of predict files are {} and number of expect files are {}".format(len(predict_files), len(expect_files)))
        print("Exiting program")
        exit()

    if predict_files != expect_files:
        print("Names of predict files and expect files do not match.")
        print("Continuing by comparing corresponding files in sorted order...")

    print("Comparing {} files...".format(len(predict_files)))

    dice_coefficient = get_dice_coefficient(predict_files, expect_files)
    if 'dice' in EVALUATE_METRICS:
        print('Dice co-efficient: ', dice_coefficient)
    if 'jaccard' in EVALUATE_METRICS:
        print('Jaccard index: ', get_jaccard_index(dice_coefficient))
    if 'accuracy' in EVALUATE_METRICS:
        print('Accuracy: ', get_accuracy(predict_files, expect_files))


def get_dice_coefficient(predict_files, expect_files, ignore_rgb_expect=True):
    dice_coefficient_sum = 0.0
    skipped_count = 0
    for predict_file, expect_file in zip(predict_files, expect_files):
        predict_img = np.asarray(Image.open(predict_file)).astype(np.bool)
        expect_img = np.asarray(Image.open(expect_file)).astype(np.bool)
        if ignore_rgb_expect:
            expect_img = expect_img[:, :, 0]
        if predict_img.shape != expect_img.shape:
            print("ERROR! Dimensions of {} and {} do not match.".format(predict_file, expect_file))
            continue
        img_sum = predict_img.sum() + expect_img.sum()
        if img_sum == 0:
            skipped_count += 1
            continue
        dice_coefficient_sum += 2 * np.sum(np.logical_and(predict_img, expect_img)) / img_sum
    return dice_coefficient_sum / (len(predict_files) - skipped_count)


def get_jaccard_index(dice_coefficient):
    return dice_coefficient / (2 - dice_coefficient)


def get_accuracy(predict_files, expect_files, ignore_rgb_expect=True):
    accuracy_sum = 0.0
    for predict_file, expect_file in zip(predict_files, expect_files):
        predict_img = np.asarray(Image.open(predict_file)).astype(np.bool)
        expect_img = np.asarray(Image.open(expect_file)).astype(np.bool)
        if ignore_rgb_expect:
            expect_img = expect_img[:, :, 0]
        if predict_img.shape != expect_img.shape:
            print("ERROR! Dimensions of {} and {} do not match.".format(predict_file, expect_file))
            continue
        true_positive = np.sum(np.logical_and(predict_img, expect_img))
        true_negative = np.sum(np.logical_and(np.logical_not(predict_img), np.logical_not(expect_img)))
        false_positive_negative = np.sum(predict_img) + np.sum(expect_img) - 2 * true_positive
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive_negative)
        accuracy_sum += accuracy
    return accuracy_sum/len(predict_files)


def get_list_of_files(dir_name):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(dir_name):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]
    return list_of_files


if __name__ == '__main__':
    main()