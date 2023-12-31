from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class MultiWindowMSG3DEncoderConfig:
    output_size: int = 16
    gcn_scales: int = 13
    g3d_scales: int = 6