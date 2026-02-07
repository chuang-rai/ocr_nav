import numpy as np


def convert_type_to_kuzu_type(value):
    if value in [int, np.int8, np.int16, np.int32, np.int64]:
        return "INT"
    elif value in [float, np.float32, np.float64]:
        return "DOUBLE"
    elif value is str:
        return "STRING"
    elif value is bool:
        return "BOOL"
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Only support fixed-length tuple of size 2")
        assert isinstance(value[1], int), "The second element of the tuple must be an integer representing the length"
        elem_type = convert_type_to_kuzu_type(value[0])
        return f"{elem_type}[{value[1]}]"
    elif isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Cannot determine the type of an empty list")
        elem_type = convert_type_to_kuzu_type(value[0])
        return f"{elem_type}[]"
    else:
        raise ValueError(f"Unsupported type: {type(value)}")
