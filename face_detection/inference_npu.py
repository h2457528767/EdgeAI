# ---------------------------------------------------------------------
# Copyright 2022-2025 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script showing how to run facenet model inference on cix npu.
"""
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union, List
from tqdm import tqdm
import sys
import argparse
import os

# Define the absolute path to the utils package by going up four directory levels from the current file location
_abs_path = os.path.join(os.path.dirname(__file__), "./")
# Append the utils package path to the system path, making it accessible for imports
sys.path.append(_abs_path)
from utils.tools import get_file_list
from utils.NOE_Engine import EngineInfer

def get_args():
    parser = argparse.ArgumentParser()
    # Argument for the path to the image or directory containing images
    parser.add_argument(
        "--image_path",
        default="./img/",
        help="path to the image file path or dir path.\
            eg. image_path=./img/",
    )
    # Argument for the path to the cix binary model file
    parser.add_argument(
        "--model_path",
        default="facenet.cix",
        help="path to the model file",
    )
    parser.add_argument(
        "--output_dir", default="./output", help="path to the result output"
    )
    args = parser.parse_args()
    return args

def smart_resize(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to a target shape while preserving aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    shape : Tuple[int, int]
        The target shape (height, width).

    Returns
    -------
    np.ndarray
        The resized image
    """

    Ht, Wt = shape
    if image.ndim == 2:
        Ho, Wo = image.shape
        Co = 1
    else:
        Ho, Wo, Co = image.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            image,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack(
            [smart_resize(image[:, :, i], shape) for i in range(Co)], axis=2
        )


class FaceLandmarkDetector:
    """
    The OpenPose face landmark detector model using ONNXRuntime.

    Parameters
    ----------
    face_model_path : str
        The path to the ONNX model file.
    """

    def __init__(self, face_model_path) -> None:
        """
        Initialize the OpenPose face landmark detector model.

        Parameters
        ----------
        face_model_path : Path
            The path to the ONNX model file.
        """

        # Initialize
        self.model = EngineInfer(face_model_path)
        self.input_name = 'input'

    def _inference(self, face_img: np.ndarray) -> np.ndarray:
        """
        Run the OpenPose face landmark detector model on an image.

        Parameters
        ----------
        face_img : np.ndarray
            The input image.

        Returns
        -------
        np.ndarray
            The detected keypoints.
        """

        # face_img should be a numpy array: H x W x C (likely RGB or BGR)
        H, W, C = face_img.shape

        # Preprocessing
        w_size = 368  # ONNX is exported for this size
        # Resize input image
        resized_img = cv2.resize(
            face_img, (w_size, w_size), interpolation=cv2.INTER_LINEAR
        )

        # Normalize: /256.0 - 0.5 (mimicking original code)
        x_data = resized_img.astype(np.float32) / 256.0 - 0.5

        # Convert to channel-first format: (C, H, W)
        x_data = np.transpose(x_data, (2, 0, 1))

        # Add batch dimension: (1, C, H, W)
        x_data = np.expand_dims(x_data, axis=0)

        # Run inference
        outputs = self.model.forward(x_data)

        # Assuming the model's last output corresponds to the heatmaps
        # and is shaped like (1, num_parts, h_out, w_out)
        heatmaps_original = outputs[-1]

        # Remove batch dimension: (num_parts, h_out, w_out)
        heatmaps_original = heatmaps_original.reshape((1, 71, 46, 46))
        heatmaps_original = heatmaps_original[0]

        # Resize the heatmaps back to the original image size
        num_parts = heatmaps_original.shape[0]
        heatmaps = np.zeros((num_parts, H, W), dtype=np.float32)
        for i in range(num_parts):
            heatmaps[i] = cv2.resize(
                heatmaps_original[i], (W, H), interpolation=cv2.INTER_LINEAR
            )

        peaks = self.compute_peaks_from_heatmaps(heatmaps)

        return peaks

    def __call__(
        self,
        face_img: Union[np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]],
    ) -> List[np.ndarray]:
        """
        Run the OpenPose face landmark detector model on an image.

        Parameters
        ----------
        face_img : Union[np.ndarray, Image.Image, List[Image.Image]]
            The input image or a list of input images.

        Returns
        -------
        List[np.ndarray]
            The detected keypoints.
        """

        if isinstance(face_img, Image.Image):
            image_list = [np.array(face_img)]
        elif isinstance(face_img, list):
            if isinstance(face_img[0], Image.Image):
                image_list = [np.array(img) for img in face_img]
            elif isinstance(face_img[0], np.ndarray):
                image_list = face_img
            else:
                raise ValueError("List elements must be PIL.Image or np.ndarray")
        elif isinstance(face_img, np.ndarray):
            if face_img.ndim == 4:
                image_list = [img for img in face_img]
            elif face_img.ndim == 3:
                image_list = [face_img]
            else:
                raise ValueError("Unsupported ndarray shape.")
        else:
            raise ValueError("Unsupported input type.")

        results = []

        for image in tqdm(image_list):
            keypoints = self._inference(image)
            results.append(keypoints)

        return results

    def compute_peaks_from_heatmaps(self, heatmaps: np.ndarray) -> np.ndarray:
        """
        Compute the peaks from the heatmaps.

        Parameters
        ----------
        heatmaps : np.ndarray
            The heatmaps.

        Returns
        -------
        np.ndarray
            The peaks, which are keypoints.
        """

        all_peaks = []
        for part in range(heatmaps.shape[0]):
            map_ori = heatmaps[part].copy()
            binary = np.ascontiguousarray(map_ori > 0.02, dtype=np.uint8)

            if np.sum(binary) == 0:
                all_peaks.append([-1, -1])
                continue

            positions = np.where(binary > 0.5)
            intensities = map_ori[positions]
            mi = np.argmax(intensities)
            y, x = positions[0][mi], positions[1][mi]
            all_peaks.append([x, y])

        return np.array(all_peaks)
    def release(self):
            self.model.clean()

if __name__ == "__main__":
    args = get_args()
    # Get a list of images from the provided path
    images_list = get_file_list(args.image_path)
    print(images_list)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    detector = FaceLandmarkDetector(args.model_path)

    for image_path in images_list:
        image = cv2.imread(image_path)
        print(image_path)
        if image is None:
            raise FileNotFoundError(f"Error can't open: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("no detect faces")
            exit(0)

        print(f"detect {len(faces)} faces")

        for i, (x, y, w, h) in enumerate(faces):
            margin = int(0.2 * w)
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, image.shape[1])
            y2 = min(y + h + margin, image.shape[0])

            face_img = image[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            keypoints_list = detector(face_rgb)
            keypoints = keypoints_list[0]

            for (px, py) in keypoints:
                if px != -1 and py != -1:
                    cv2.circle(face_img, (int(px), int(py)), 2, (0, 255, 0), -1)

            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            out_image_path = os.path.join(
                output_dir, "npu_" + os.path.basename(image_path)
                )

            cv2.imwrite(out_image_path, image)

    detector.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
