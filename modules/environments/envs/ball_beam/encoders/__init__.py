from .dynamic_x_y import dynamic_encoding_x_y, phase_mapping_x_y
from .dynamic_degree import dynamic_encoding_degree, phase_mapping_degree

ENCODERS = {
    "Dynamic Encoding XY": {
        "encoder": dynamic_encoding_x_y,
        "phase_map": phase_mapping_x_y,
        "encode_count": 5,
    },
    "Dynamic Encoding Degree": {
        "encoder": dynamic_encoding_degree,
        "phase_map": phase_mapping_degree,
        "encode_count": 5,
    },
}
