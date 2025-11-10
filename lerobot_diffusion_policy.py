from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from torchvision import models, transforms
from torchvision.transforms import ToPILImage

# Login using e.g. `huggingface-cli login` to access this dataset
ds = LeRobotDataset(repo_id = "ETHRC/ethrc_piper_screw_driver")
#print(ds[0].keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_directory = Path("/Users/hallyhu/Documents/RL-100")

dataset_metadata = LeRobotDatasetMetadata("ETHRC/ethrc_piper_screw_driver")
features = dataset_to_policy_features(dataset_metadata.features)
#print("Features:", features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = DiffusionConfig(n_obs_steps=2, vision_backbone="resnet18", crop_shape=[224, 224], input_features=input_features, output_features=output_features, noise_scheduler_type="DDIM")
policy = DiffusionPolicy(cfg)
preprocessor, postprocessor = make_pre_post_processors(
    cfg, dataset_stats=dataset_metadata.stats
)

policy.train()
policy.to(device)


# Create the optimizer and dataloader for offline training
optimizer = cfg.get_optimizer_preset().build(policy.parameters())
batch_size = 32
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# Number of training steps and logging frequency
training_steps = 1
log_freq = 1

# Run training loop
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = preprocessor(batch)

        #need to adjust batch to have correct shape
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break



""""
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize shorter side to 256
    transforms.CenterCrop(224),            # Center crop to 224x224
    transforms.ToTensor(),                 # Convert PIL image to PyTorch tensor [C,H,W], values 0-1
])
"""