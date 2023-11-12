import os
import cv2
from skimage import exposure

input_dir = "dataset/output/train"
output_dir = "dataset/output/filtered/train"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
target_size = (224, 224)

for image_file in image_files:
    img_rgb = cv2.imread(os.path.join(input_dir, image_file))

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    green_channel = img_bgr[:, :, 1]
    cv2.imshow('Green Channel Image', green_channel)

    complement_green_channel = 255 - green_channel
    cv2.imshow('Complement of Green Channel Image', complement_green_channel)

    clahe_complement = exposure.equalize_adapthist(complement_green_channel, clip_limit=0.03)
    clahe_complement = cv2.normalize(clahe_complement, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)

    cv2.imwrite(output_dir + '/' + image_file, clahe_complement)
    cv2.imshow('Original Image', img_rgb)
    cv2.imshow('Equalized Image with Complemented Green Channel', clahe_complement)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
