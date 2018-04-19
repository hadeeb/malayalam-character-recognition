import os
import cv2
import numpy as np

IMAGE_SIZE = 32


def list_folders(root_folder):
    """Function to get subdir list"""
    folder_list = []
    for folder in sorted(os.listdir(root_folder)):
        if os.path.isdir(os.path.join(root_folder, folder)):
            folder_list.append(folder)
    return folder_list


def create_folders(root_folder, folder_list):
    """Function to create folders in new dataset"""
    for folder in folder_list:
        os.makedirs(os.path.join(root_folder, folder), exist_ok=True)


def read_transparent_png(filename):
    """
    Change transparent bg to white
    """
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def clean(img):
    """Process an image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (__, img_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    __, ctrs, __ = cv2.findContours(img_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # take largest contour
    ctr = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[2] * cv2.boundingRect(ctr)[3]),
                 reverse=True)[0]
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img_bw[y:y + h, x:x + w]
    return skeletize(crop(roi, IMAGE_SIZE))


def crop(image, desired_size):
    """Crop and pad to req size"""
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def process_folder(folder):
    """Process all images in a folder"""
    extension = '.png'
    new_list = []
    for img in sorted(os.listdir(folder)):
        if img.endswith(extension):
            try:
                image = read_transparent_png(os.path.join(folder, img))
                new_img = clean(image)
                new_list.append([img, new_img])
            except:
                print("\t" + img)
    return new_list


def save_new(folder, imglist):
    """Save newly created images"""
    for img in imglist:
        cv2.imwrite(os.path.join(folder, img[0]), img[1])


def process_images(raw_folder, clean_folder, folder_list):
    """Process the images"""
    for folder in folder_list:
        print(folder)
        imglist = process_folder(os.path.join(raw_folder, folder, 'output'))
        save_new(os.path.join(clean_folder, folder), imglist)


def skeletize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeroes = size - cv2.countNonZero(img)
        if zeroes == size:
            done = True

    return skel
