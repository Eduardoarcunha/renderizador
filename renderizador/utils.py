import math
import numpy as np


def downsample_matrix_with_channels(input_matrix, factor=2):
    rows, cols, channels = input_matrix.shape

    # Ensure rows and columns are divisible by the factor, trim if necessary
    if rows % factor != 0:
        input_matrix = input_matrix[: rows - (rows % factor), :, :]
    if cols % factor != 0:
        input_matrix = input_matrix[:, : cols - (cols % factor), :]

    # Reshape the matrix into blocks of size factor x factor for each channel and calculate the mean
    downsampled = np.mean(
        input_matrix.reshape(rows // factor, factor, cols // factor, factor, channels),
        axis=(1, 3),
    )

    return downsampled


def normalize_vector(v):
    magnitude = np.linalg.norm(v)
    if magnitude != 0:
        v = v / magnitude
    return v


def vector_module(v):
    return math.sqrt(sum([x**2 for x in v]))
