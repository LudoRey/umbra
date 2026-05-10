from collections.abc import Callable

import numpy as np

ImageCallback = Callable[[np.ndarray], None]
CheckStateCallback = Callable[[], None]
