from .dynamic import dynamic_encoding, phase_mapping

ENCODERS = {
    "Dynamic Encoding": {
        "encoder": dynamic_encoding,
        "phase_map": phase_mapping,
        "encode_count": 4
    }
}