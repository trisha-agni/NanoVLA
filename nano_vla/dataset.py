# external imports
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
import json
import os
import torch
# internal imports
from nano_vla.config import DEFAULT_MODEL_ID, MANIFEST_PATH

MANIFEST_JSON_NAME = "manifest.json"
MAX_TOKEN_LENGTH = 32
STD_IMAGENT_SCALING = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
VISION_ENCODER_RES = (224, 224)  # standard input resolution for typical vision encoder


class NanoVLADataset(Dataset):
    def __init__(self, model_id=DEFAULT_MODEL_ID):
        """Loads the manifest metadata and prepares the vision/language processors."""
        self.samples = self._load_manifest()
        self.tokenizer = self._create_tokenizer(model_id)
        self.img_transform = self._create_vision_transformer()

    def _load_manifest(self):
        """Load the generated recording logs."""
        if not os.path.exists(MANIFEST_PATH):
            raise FileNotFoundError(f"No manifest.json at {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r") as f:
            return json.loads(f)

    def _create_tokenizer(self):
        """
        Set up the text tokenizer
        (using lightweight gpt-2 as a proxy backbone).
        """
        tok = AutoTokenizer.from_pretrained(self.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _create_vision_transformer(self):
        """
        Set up the vision transformers
        (converts PNGs to standardized torch float matrices).
        """
        return transforms.Compose([
            transforms.Resize(VISION_ENCODER_RES),
            transforms.ToTensor(),  # scales pixel values from [0, 255] to [0.0, 1.0]
            transforms.Normalize(
                STD_IMAGENT_SCALING['mean'],
                STD_IMAGENT_SCALING['std']
            )
        ])

    def __len__(self):
        """Returns the total no. of recorded frames in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetches a single training sample, processing its image, text and labels."""
        sample = self.samples[idx]

        # vision loading
        img_raw = Image.open(sample['image_path']).convert('RGB')
        img_tensor = self.img_transform(img_raw)

        # language tokens
        text_encoded = self.tokenizer(
            sample['instruction'],
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt'
        )

        # action handling
        action_label = torch.tensor(sample['action_token_id'], dtype=torch.long)

        # vision tensor: [3, 224, 224]
        # language tokens: [32]
        # language mask: [32]
        # target optimization token id
        return {
            'pixel_values': img_tensor,
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'labels': action_label
        }


if __name__ == "__main__":
    # test your pipeline on your newly recorded dataset folder
    try:
        dataset = NanoVLADataset()
        print("📊 Dataset verification successful!")
        print(f"Total recorded samples found: {len(dataset)}")

        # pull the very first item out of the loader pipeline
        first_sample = dataset[0]
        print("\n⚡️ Tensor Structural Verification:")
        print(f" -> Pixel values shape (Vision): {first_sample['pixel_values'].shape}")
        print(f" -> Input Token IDs shape (Language): {first_sample['input_ids'].shape}")
        print(f" -> Target Action Token ID (Behavior): {first_sample['labels'].item()}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
