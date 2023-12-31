import torch

from transformers import AutoConfig, Wav2Vec2Model, Wav2Vec2Processor

from layers.config.wav2vecencoder import Wav2VecEncoderConfig, Pooling

class Wav2VecEncoder(torch.nn.Module):
    def __init__(self, config: Wav2VecEncoderConfig) -> None:
        super().__init__()

        self.config = config

        classifier_config = AutoConfig.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", finetuning_task="wav2vec2_clf")
        classifier_config.layerdrop = 0

        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.classifier = Wav2Vec2Model(classifier_config)

        self.dropout = torch.nn.Dropout(p=Wav2VecEncoderConfig().dropout)
        self.norm = torch.nn.LayerNorm(self.classifier.config.hidden_size)
        self.output = torch.nn.Linear(self.classifier.config.hidden_size, Wav2VecEncoderConfig().output_size)

    def _merge(self, hidden):
        if self.config.pooling == Pooling.MEAN:
            return hidden.mean(dim=1)
        elif self.config.pooling == Pooling.SUM:
            return hidden.sum(dim=1)
        elif self.config.pooling == Pooling.MAX:
            return hidden.max(dim=1)[0]

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        processed = self.processor([audio[i][:lengths[i]][:, 0].cpu().numpy() for i in range(audio.shape[0])], sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True)

        processed['input_values'] = processed['input_values'].to(audio.device)
        processed['attention_mask'] = processed['attention_mask'].to(audio.device)

        x = self.classifier(**processed).last_hidden_state

        x = self.dropout(x)
        x = self.norm(self._merge(x))
        x = self.output(x)

        return torch.tanh(x)