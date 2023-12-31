import enum

import opensmile

from dlpipeline.config.config import configuration_property

class AudioFeatureExtractor(enum.Enum):
    OPENSMILE = 'opensmile'
    PASST = 'passt'

@configuration_property(prefix='experiment.data.transform.config.extract_audio_features')
class ExtractAudioFeaturesConfig:
    extractor: AudioFeatureExtractor = AudioFeatureExtractor.OPENSMILE

    # OpenSmile
    opensmile_featureset: opensmile.FeatureSet = opensmile.FeatureSet.ComParE_2016
    opensmile_featurelevel: opensmile.FeatureLevel = opensmile.FeatureLevel.Functionals