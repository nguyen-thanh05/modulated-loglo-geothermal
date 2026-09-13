import torch


class SingleTensorAdapter:
    """Concatenate state, static fields, rate, and well mask into one tensor."""

    def build_model_input(self, y, action, static):
        rate = action.unsqueeze(1)
        mask = (rate != 0).float()
        return torch.cat([y, static, rate, mask], dim=1)

    def forward(self, model, model_input):
        return model(model_input)


class DualTensorAdapter:
    """Modulated LOGLO: spatial state+static, separate action+static modulation."""

    def build_model_input(self, y, action, static):
        rate = action.unsqueeze(1)
        mask = (rate != 0).float()
        state_input = torch.cat([y, static], dim=1)
        action_input = torch.cat([rate, mask, static], dim=1)
        return (state_input, action_input)

    def forward(self, model, model_input):
        state_input, action_input = model_input
        return model(state_input, action_input)


def split_model_output(output):
    """Return (spatial, aux_or_none) from a model that may emit aux."""
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    return output, None


def create_adapter(model_type):
    if model_type in ('modulated_loglo', 'modulated_loglo_aux'):
        return DualTensorAdapter()
    if model_type in ('unet3d', 'fno', 'ufno', 'vanilla_loglo'):
        return SingleTensorAdapter()
    raise ValueError(f"Unknown model_type: {model_type}")
