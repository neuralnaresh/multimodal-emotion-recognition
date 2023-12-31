import torch
import torch_scatter

class DCRNReason(torch.nn.Module):
    def __init__(self, input_size: int, steps: int, layers: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size * 2
        self.steps = steps
        self.layers = layers

        if steps > 0:
            self.lstm = torch.nn.LSTM(self.output_size, self.input_size, layers)

    def forward(self, input: torch.Tensor, batch: torch.Tensor, qstar: torch.Tensor) -> torch.Tensor:
        if self.steps <= 0:
            return qstar

        batch_size = batch.max().item() + 1

        h = (input.new_zeros((self.num_layers, batch_size, self.in_channels)),
             input.new_zeros((self.num_layers, batch_size, self.in_channels)))

        for _ in range(self.processing_steps):
            q, h = self.lstm(qstar.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)

            e = (input * q[batch]).sum(dim=-1, keepdim=True)
            a = torch.softmax(e, batch, num_nodes=batch_size)
            r = torch_scatter.scatter_add(a * input, batch, dim=0, dim_size=batch_size)

            qstar = torch.cat([q, r], dim=-1)

        return qstar

        
    