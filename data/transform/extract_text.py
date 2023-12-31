import typing

import torch

import librosa
import transformers

from dlpipeline.data.transform import Transformer
from dlpipeline.commands.experiment_config import ExperimentConfig

from data.transform.config.extract_text import ExtractTextConfig

import data.constants
import data.utils


class ExtractText(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.processor = transformers.Wav2Vec2Processor.from_pretrained(
            ExtractTextConfig().processor
        )
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained(
            ExtractTextConfig().model
        ).to(ExperimentConfig().device)

    def transcribe(self, file: str) -> str:
        waveform, _ = librosa.load(file, sr=16000)
        waveform = torch.tensor(waveform)

        input = self.processor(waveform, return_tensors="pt", padding="longest", sampling_rate=16000).input_values

        self.model.eval()
        logits = self.model(input.to(ExperimentConfig().device)).logits

        predictions = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predictions)[0]
        
        return transcription

    def transform_single(
        self, input_filename: str, output_base_dir: str
    ) -> typing.Dict[str, typing.Any]:
        data_sample = data.utils.read_pickle(input_filename)

        data_sample[data.constants.DATA_TEXT] = self.transcribe(
            data_sample[data.constants.DATA_AUDIO_PATH]
        )

        return data_sample

    def write_single(
        self,
        processed: typing.Dict[str, typing.Any],
        input_filename: str,
        output_base_dir: str,
    ) -> None:
        return data.utils.write_transformer_pickle(
            processed, input_filename, output_base_dir
        )
