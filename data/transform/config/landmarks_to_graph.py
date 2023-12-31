from dlpipeline.config.config import configuration_property

@configuration_property(prefix='experiment.data.transform.config.landmarks_to_graph')
class LandmarksToGraphConfig:
    convert_all_faces: bool = False
    convert_mocap: bool = False
    edges_first: bool = False
    frames: int = 4