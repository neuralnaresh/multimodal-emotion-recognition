from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.tranform.config.extract_faces')
class ExtractFacesConfig:
    buffer_size: int = 5