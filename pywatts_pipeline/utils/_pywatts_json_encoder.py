import json
import numpy as np


class PyWATTSJsonEncoder(json.JSONEncoder):
    """
    Special json encoder for types that are not json-serializable by default.
    """

    def default(self, o):
        # numpy encoding
        # reference: https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()

        return json.JSONEncoder.default(self, o)
