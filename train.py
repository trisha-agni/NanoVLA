# external imports
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
# internal imports
from nano_vla.dataset import NanoVLADataset
from nano_vla.model import NanoVLAModel

BATCH_SZ = 4
DEFAULT_LR = 1e-4
EPOCHS = 5
WEIGHTS_FILE = 'nanovla_checkpoint.pt'


def train_nanovla(epochs=EPOCHS, batch_sz=BATCH_SZ, lr=DEFAULT_LR):
    print("🚀 Initializing NanoVLA Training Pipeline...")

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f' using hardware acceleration: [{device}]')

    # instantiate dataset and data loader
    try:
        dataset = NanoVLADataset()
        dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
        print(f'📊 Dataset loaded successfully!'
              f'Total training pairs discovered: {len(dataset)}')
    except Exception as e:
        print(f'failed to read data: {e}')
        return

    # instantiate the model and shift it to our compute hardware
    model = NanoVLAModel()
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    print('\nStarting optimization loop...')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # send our chunked multimodal tensors over to the compute device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # clear historical gradient tracks before processing step calculations
            optimizer.zero_grad()

            # forward pass: compute predictive action probabilities
            action_logits = model(pixel_values=pixel_values,
                                  input_ids=input_ids)

            # loss computation
            loss = criterion(action_logits, labels)

            # backward pass: calculate parameter updates using gradient calculus
            loss.backward()

            # optimization step: tweak weights to minimize driving errors
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f' |- Epoch [{epoch+1:02d}/{epochs}] --> '
              f'Average Imitation Loss: {avg_loss:.4f}')

    print('\n🏆 NanoVLA training complete! Saving model checkpoint parameters...')
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"Saved weight securely to '{WEIGHTS_FILE}'")


if __name__ == '__main__':
    train_nanovla(epochs=EPOCHS, batch_sz=BATCH_SZ)
