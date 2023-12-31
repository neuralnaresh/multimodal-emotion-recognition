from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.transform.config.extract_pyfeat')
class ExtractPyFeatConfig:
    face_model: str = 'mtcnn'
    landmark_model: str = 'pfld'
    au_model: str = 'jaanet'
    emotion_model: str = 'resmasknet'
    facepose_model: str = 'img2pose'
    batch_size: int = 16