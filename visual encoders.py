from lerobot.datasets.lerobot_dataset import LeRobotDataset  
#from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransfromConfig
import torch
from torchvision import models, transforms
from torchvision.transforms import ToPILImage

# Login using e.g. `huggingface-cli login` to access this dataset
ds = LeRobotDataset(repo_id = "ETHRC/ethrc_piper_screw_driver")
print(ds[0].keys())
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize shorter side to 256
    transforms.CenterCrop(224),            # Center crop to 224x224
    transforms.ToTensor(),                 # Convert PIL image to PyTorch tensor [C,H,W], values 0-1
])
#to_pil = ToPILImage()
#img = to_pil(ds[0]['observation.images.wrist1'])  # converts [C,H,W] float tensor to PIL

#img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension [1,C,H,W]

#img = to_pil(img_tensor.squeeze(0))  # Convert back to PIL for visualization if needed
#img.show()
#print(ds[0])
resnet = models.resnet18(pretrained=True)
resnet.eval()  # Set to evaluation mode
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

def feature_embedding(image_tensor):
    to_pil = ToPILImage()
    img = to_pil(image_tensor)  # Convert back to PIL for preprocessing
    img_tensor = preprocess(img).unsqueeze(0)  # Preprocess and add batch dimension
    
    with torch.no_grad():
        features = resnet_feature_extractor(img_tensor)
        features = features.view(features.size(0), -1)  # Flatten
    return features

emb = feature_embedding(ds[0]['observation.images.wrist1'])

def conditioning_vector(ds):
    embeddings_w1 = []
    embeddings_w2 = []
    embeddings_stereo = []
    for i in range(len(ds)):
        img_tensor_w1 = ds[i]['observation.images.wrist1']
        img_tensor_w2 = ds[i]['observation.images.wrist2']
        img_tensor_stereo = ds[i]['observation.images.stereo_left']
        emb_w1 = feature_embedding(img_tensor_w1)
        emb_w2 = feature_embedding(img_tensor_w2)
        emb_stereo = feature_embedding(img_tensor_stereo)
        embeddings_w1.append(emb_w1)
        embeddings_w2.append(emb_w2)
        embeddings_stereo.append(emb_stereo)
    return {"wrist_1": torch.cat(embeddings_w1, dim=0),
            "wrist_2": torch.cat(embeddings_w2, dim=0),
            "stereo_left": torch.cat(embeddings_stereo, dim=0)}


#print(emb.shape)