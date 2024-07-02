import skimage.util
from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure, color
from skimage.filters import try_all_threshold, threshold_minimum
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu

BINARY_THRESHOLD = 100
BOTTOM_CROP_VALUE = 225


def plot_histogram(image: Image):
    image = np.array(image)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Calculate histogram
    hist, bins = exposure.histogram(image)

    # Plot histogram
    plt.figure()
    plt.bar(bins, hist, width=0.8)
    plt.title('Histogram of Grayscale Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.show()


def adjust_exposure(image: Image, exposure_factor: float):
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(exposure_factor)

    return adjusted_image


def apply_minimum_threshold(image: Image):
    img_arr = np.array(image)
    thresh = threshold_minimum(img_arr)
    threshold_image = image > thresh

    return Image.fromarray(threshold_image)


def try_all_thresholds(image: Image):
    try_all_threshold(np.array(image), figsize=(10, 8), verbose=False)
    plt.show()


def normalize_image(image: Image):
    img_array = np.array(image)

    smoothed_img = gaussian_filter(img_array, sigma=2)
    equalized_img = exposure.equalize_adapthist(smoothed_img, clip_limit=0.03)

    return Image.fromarray((equalized_img * 255).astype(np.uint8))


def crop(img: Image, pixels_to_crop: int):
    width, height = img.size
    box = (0, 0, width, height - pixels_to_crop)
    return img.crop(box)


def prepare_image_for_area_calcs(image: Image):
    image = crop(image, BOTTOM_CROP_VALUE)
    image = normalize_image(image)
    image = adjust_exposure(image, 1.2)
    image = apply_minimum_threshold(image)
    return image


def count_unoccupied(image: Image):
    return np.count_nonzero(np.array(image))


def count_occupied(image: Image):
    arr = np.array(image)
    return arr.size - np.count_nonzero(arr)


def find_occupied_percentage(image: Image):
    return count_occupied(image) / np.array(image).size


def find_unoccupied_percentage(image: Image):
    return count_unoccupied(image) / np.array(image).size


def calculate_occupied_percentage(image: Image):
    image = prepare_image_for_area_calcs(image)
    return find_occupied_percentage(image)


def calculate_unoccupied_percentage(image: Image):
    image = prepare_image_for_area_calcs(image)
    return find_unoccupied_percentage(image)


if __name__ == '__main__':

    images = []
    for i in range(4):
        images.append(Image.open("images/IMC100nm_1As/filnm_15minanneal" + str(i) + ".tif"))

    for i, image in enumerate(images, 0):
        occupied_percentage = calculate_occupied_percentage(image)
        unoccupied_percentage = calculate_unoccupied_percentage(image)
        print(f"image {i}", occupied_percentage * 100)
        assert occupied_percentage + unoccupied_percentage == 1  # assert occupied_percentage + unoccupied_percentage == 100

    # img = Image.open("images/IMC100nm_1As/filnm_30minanneal4.tif").convert("L")
    #
    # occupied_percentage = calculate_occupied_percentage(img)
    # unoccupied_percentage = calculate_unoccupied_percentage(img)
    # print(occupied_percentage)
    # print(unoccupied_percentage)
    # assert occupied_percentage + unoccupied_percentage == 1
    # find_occupied_percentage(img) +

    # crop(img, BOTTOM_CROP_VALUE)

    # find_occupied_percentage(img)
    # try_all_thresholds(
    #     normalize_image(adjust_exposure(img, 1.5)))

    # plt.imshow(np.array(apply_minimum_threshold(normalize_image(adjust_exposure(img, 1.2)))), cmap='gray', vmin=0, vmax=1)
    # plt.show()

    # plt.show()

    # find_occupied_percentage(img)
