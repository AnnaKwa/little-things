from dataclasses import dataclass
import numpy as np


@dataclass
class Galaxy:
    name: str
    distance: float
    vlos_2d_data: np.ndarray
    deg_per_pixel: float
    age: float=None

    def __post_init__(self):
        self.image_xdim, self.image_ydim = vlos_2d_data.shape
