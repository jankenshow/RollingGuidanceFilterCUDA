import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

# import pytest

try:
    import rgf_cuda
except ImportError:
    rgf_cuda = None


def test_rgf_identity():
    if rgf_cuda is None:
        return
    img = cv2.imread(os.path.join(BASE_DIR, "test/data/dog.jpg"))
    out = rgf_cuda.gaussian_blur(img, sigma=3.0)
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/dog_gauss.jpg"), out.astype(np.uint8))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(BASE_DIR, "test/data/dog_gray.jpg"), gray)
    out_gray = rgf_cuda.gaussian_blur(gray, sigma=3.0)
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/dog_gray_gauss.jpg"), out_gray.astype(np.uint8))

    img = cv2.imread(os.path.join(BASE_DIR, "test/data/image.png"))
    start_time = time.time()
    out = rgf_cuda.gaussian_blur(img, sigma=3.0)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    cv2.imwrite(os.path.join(BASE_DIR, "test/out/image_gauss.png"), out.astype(np.uint8))


if __name__ == "__main__":
    if rgf_cuda is None:
        print("rgf_cuda not built")
    else:
        test_rgf_identity()
        print("Test passed.")
