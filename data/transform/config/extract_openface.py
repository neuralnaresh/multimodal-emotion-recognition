import enum

from dlpipeline.config.config import configuration_property

class OpenFaceExtractionType(enum.Enum):
    SINGLE = 'single'
    MULTI = 'multi'

@configuration_property(prefix='experiment.data.tranform.config.extract_frames')
class ExtractOpenFaceConfig:
    extraction_type: OpenFaceExtractionType = OpenFaceExtractionType.SINGLE