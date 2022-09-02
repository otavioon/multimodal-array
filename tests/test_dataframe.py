import pytest

import numpy as np
import pandas as pd
from multimodal.dataframe import MultiModalDataframe

def generate_dataframe(shape):
    return pd.DataFrame(
        np.arange(np.prod(shape)).reshape(shape),
        columns=[f"col-{i}" for i in range(shape[-1])]
    )
