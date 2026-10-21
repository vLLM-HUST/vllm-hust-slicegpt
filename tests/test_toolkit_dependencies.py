"""Exercise real Arrow/NumPy/datasets interoperability without downloaded data."""

import numpy as np
import pyarrow as pa
from datasets import Dataset, load_from_disk


def test_arrow_numpy_dataset_roundtrip(tmp_path):
    values = np.array([1, 4, 9], dtype=np.int64)
    table = pa.table({"value": values})
    np.testing.assert_array_equal(table["value"].to_numpy(), values)
    dataset = Dataset(table)
    destination = tmp_path / "synthetic-dataset"
    dataset.save_to_disk(destination)
    restored = load_from_disk(destination)
    np.testing.assert_array_equal(restored.with_format("numpy")["value"], values)
