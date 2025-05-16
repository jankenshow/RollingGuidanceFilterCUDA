import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "build"))

import cv2
import numpy as np

# import pytest

try:
    import rgf_pybind
except ImportError:
    rgf_pybind = None


def test_rgf_identity():
    if rgf_pybind is None:
        return
    img = cv2.imread(os.path.join(BASE_DIR, "test/data/dog.jpg"))
    out = rgf_pybind.rolling_guidance_filter(img, sigma_s=3.0, sigma_r=10.0, iterations=3)
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/dog_rgf.jpg"), out.astype(np.uint8))

    gray = cv2.imread(os.path.join(BASE_DIR, "test/data/dog_gray.jpg"))
    out_gray = rgf_pybind.rolling_guidance_filter(gray, sigma_s=3.0, sigma_r=10.0, iterations=3)
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/dog_gray_rgf.jpg"), out_gray.astype(np.uint8))

    img = cv2.imread(os.path.join(BASE_DIR, "test/data/image.png"))
    start_time = time.time()
    out = rgf_pybind.rolling_guidance_filter(img, sigma_s=3.0, sigma_r=25.0, iterations=4)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/image_rgf.png"), out.astype(np.uint8))


if __name__ == "__main__":
    if rgf_pybind is None:
        print("rgf_pybind not built")
    else:
        test_rgf_identity()
        print("Test passed.")
