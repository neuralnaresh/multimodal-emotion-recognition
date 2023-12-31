from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.tranform.config.extract_text')
class ExtractTextConfig:
    processor: str = "facebook/wav2vec2-base-960h"
    model: str = "facebook/wav2vec2-base-960h"