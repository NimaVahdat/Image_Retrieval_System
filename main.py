import timm 
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision



# =============================================================================
# Resources:
# https://github.com/OlgaChernytska/word2vec-pytorch
# https://timm.fast.ai/
# https://github.com/huggingface/pytorch-image-models
# https://pytorch.org/vision/0.8/models.html  
# https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
# =============================================================================

folder = "./word2vec/weights/cbow_WikiText2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")


descriptions = [
    'a fish in a lake',
    'the best professor',
    'an fruit',
    'two big trees',
    'throwing a ball',
    'party time',
    'red star',
    'cat with a cat',
    'guitar on television',
    'fox in a box',
]

class Projector(nn.Module):
    def __init__(self):
        super(Projector,self).__init__()
        self.layers = nn.Sequential(
                                    nn.Linear(512,300),
                                    nn.ReLU(),
                                    nn.Linear(300,300)
        )
    def forward(self,x):
        return self.layers(x)

class Haystack(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.all_ims = np.load('haystack.npy')
        np.random.seed(300)
        self.personalized_idxs = np.random.permutation(10000)

    def __len__(self):
        return 10000
    def __getitem__(self,idx):
        personalized_idx = self.personalized_idxs[idx]
        im = self.all_ims[personalized_idx,...]
        im = torch.Tensor(im)/255
        im = torchvision.transforms.functional.resize(im,[224,224])
        return im

def cosine_similarity(input1, input2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    return cos(input1, input2)


dataset = Haystack()

vgg16_model = timm.create_model('vgg16', pretrained=True)
vgg16_model = vgg16_model.eval()

# vgg16_model_features = nn.Sequential(*list(vgg16_model.children())[:-2])

image_embeddings = []
for idx in range(len(dataset)):
    image = dataset[idx]
    with torch.no_grad():
        image_features = vgg16_model.forward_features(image)
        image_embedding = torch.mean(image_features, dim=[1, 2]).squeeze(0)
        image_embeddings.append(image_embedding)



embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
print(embeddings_norm.shape)

description_embeddings = []
for description in descriptions:
        words =  description.split()
        em = []
        for word in words:
                description_embedding = vocab[word]
                description_embedding = embeddings_norm[description_embedding]
                em.append(description_embedding)
        description_embeddings.append(np.mean(em, axis=0))

projection_network = Projector()
projection_network.load_state_dict(torch.load('projector.pth'))
projection_network.eval()
projected_image_embeddings = torch.zeros((10000, 300))
i = 0
for image_embedding in image_embeddings:
    with torch.no_grad():
        projected_image_embedding = projection_network(image_embedding)
        # projected_image_embeddings.append(projected_image_embedding)
        projected_image_embeddings[i] = projected_image_embedding
        i += 1

for i, description_embedding in enumerate(description_embeddings):
    similarities = []
    for j, projected_image_embedding in enumerate(projected_image_embeddings):
        similarity = cosine_similarity(torch.tensor(description_embedding), projected_image_embedding)
        # print(similarity)
        similarities.append(similarity)

    max_similarity_idx = similarities.index(max(similarities))
    matched_image = dataset[max_similarity_idx]
    matched_image = matched_image.numpy()
    matched_image *= 255
    matched_image = matched_image.astype('uint8')
    image = Image.fromarray(matched_image.transpose(1, 2, 0))
    image.save(f"matched_image_{i}.jpg")
    print(f"Matched image {i} is located at index {max_similarity_idx}.")

