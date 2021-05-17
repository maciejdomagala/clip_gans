import torch
from pytorch_pretrained_biggan import BigGAN
from clip import clip
import kornia
from PIL import Image
from torchvision.utils import save_image


from utils import save_grid, freeze_model


class Generator:
    def __init__(self, config):
        self.config = config
        self.augmentation = None

        self.CLIP, clip_preprocess = clip.load(
            "ViT-B/32", device=self.config.device, jit=False)
        self.CLIP = self.CLIP.eval()
        freeze_model(self.CLIP)
        self.model = self.config.model(config).to(self.config.device).eval()
        freeze_model(self.model)

        self.tokens = clip.tokenize(
            [self.config.target]).to(self.config.device)
        self.text_features = self.CLIP.encode_text(self.tokens).detach()
        # if config.task == "img2txt":
        #     image = clip_preprocess(Image.open(self.config.target)).unsqueeze(
        #         0).to(self.config.device)
        #     self.image_features = self.CLIP.encode_image(image)

    def generate(self, ls, minibatch=None):
        z = ls()
        result = self.model.generate(*z, minibatch=minibatch)
        if hasattr(self.config, "norm"):
            result = self.config.norm(result)
        return result

    def discriminate(self, images, minibatch=None):
        images = self.config.denorm(images)
        return self.model.discriminate(images, minibatch)

    def has_discriminator(self):
        return self.model.has_discriminator()

    def clip_similarity(self, input):
        image = kornia.resize(input, (224, 224))
        if self.augmentation is not None:
            image = self.augmentation(image)

        image_features = self.CLIP.encode_image(image)

        sim = torch.cosine_similarity(image_features, self.text_features)

        return sim

    def save(self, input, path):
        if input.shape[0] > 1:
            save_grid(input.detach().cpu(), path)
        else:
            save_image(input[0], path)
