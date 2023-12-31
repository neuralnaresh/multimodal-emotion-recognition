import dataclasses
import typing

from dlpipeline.config.config import nested_configuration_property

@nested_configuration_property
class DCRNCognitionConfig:
    dropout: float = 0.2
    steps: typing.Union[typing.List[int], None] = None
