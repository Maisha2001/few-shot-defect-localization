import unittest
from PIL import Image
import torch
from transformers import Dinov2Config, Dinov2Model

from inspection.model import embeddings, pixels

class EncoderTests(unittest.TestCase):
    def test_real_encoder_discards_cls_and_normalizes_local_features(self):
        torch.set_num_threads(2)
        model = Dinov2Model(Dinov2Config(image_size=224, patch_size=14, hidden_size=384,
                                         num_hidden_layers=1, num_attention_heads=6)).eval()
        inputs = [pixels(Image.new("RGB", (25,40), (128,80,30)))]
        encoded = embeddings(model, inputs)
        self.assertEqual(encoded.shape, (1,256,384))
        torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(1,256))
        with torch.inference_mode():
            raw = model(pixel_values=torch.stack(inputs)).last_hidden_state
        torch.testing.assert_close(encoded[0,0], raw[0,1] / raw[0,1].norm())
