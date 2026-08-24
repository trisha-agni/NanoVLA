# external imports
from transformers import AutoModel, AutoModelForCasualLM
import torch
import torch.nn as nn
# internal imports
from nano_vla.config import DEFAULT_MODEL_ID, DEFAULT_VISION_ENCODER


class NanoVLAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # create language model (the brain)
        self.language_model, self.lm_hidden_dim = self._create_lang_model()

        # create vision encoder (the eyes)
        self.vision_encoder, self.vision_hidden_dim = self._create_vision_encoder()

        # the MLP projector bridge
        # this bridges the spatial dimension gap between what the eyes see
        # and what the brain reads
        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_dim, self.lm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.lm_hidden_dim, self.lm_hidden_dim)
        )

        def forward(self, pixel_values, input_ids, attention_mask=None):
            """
            Executes a single forward multimodal training pass.
            Sequence construction:
            [Vision Tokens] + [Text Instruction Tokens] -> Predict Action Token
            """
            batch_sz = pixel_values.shape[0]

    def _create_lang_model(self):
        """
        Create language model (the brain).
        By default, we use standard causal GPT-2.
        """
        obj = AutoModelForCasualLM.from_pretrained(DEFAULT_MODEL_ID)
        # gpt-2 hidden size is usually 768
        return obj, obj.config.n_embd

    def _create_vision_encoder(self):
        """
        Create vision encoder (the eyes).
        By default, we use a tiny pretrained standard Vision Transformer (ViT)
        from HuggingFace.
        """
        obj = AutoModel.from_pretrained(DEFAULT_VISION_ENCODER)
        for param in obj.parameters():
            # freeze the vision encoder weights (standard VLA practice
            # to preserve pretrained features)
            param.requires_grad = False
        # ViT hidden size is usually 768
        return obj, obj.config.hidden_size