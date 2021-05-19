import os
import sys
import torch
from pytorch_pretrained_biggan import BigGAN as DMBigGAN
import stylegan2


class DeepMindBigGAN(torch.nn.Module):
    def __init__(self, config):
        super(DeepMindBigGAN, self).__init__()
        self.config = config
        self.G = DMBigGAN.from_pretrained(config.weights)
        self.D = None

    def has_discriminator(self):
        return False

    def generate(self, z, class_labels, minibatch=None):
        if minibatch is None:
            return self.G(z, class_labels, self.config.truncation)
        else:
            assert z.shape[0] % minibatch == 0
            gen_images = []
            for i in range(0, z.shape[0] // minibatch):
                z_minibatch = z[i*minibatch:(i+1)*minibatch, :]
                cl_minibatch = class_labels[i*minibatch:(i+1)*minibatch, :]
                gen_images.append(
                    self.G(z_minibatch, cl_minibatch, self.config.truncation))
            gen_images = torch.cat(gen_images)
            return gen_images


class StyleGAN2(torch.nn.Module):
    def __init__(self, config):
        super(StyleGAN2, self).__init__()
        if not os.path.exists(os.path.join(config.weights, "G.pth")):
            if "ffhq" in config.config:
                model = "ffhq"
            elif "car" in config.config:
                model = "car"
            elif "church" in config.config:
                model = "church"
            elif "cat" in config.config:
                model = "cat"
            elif "horse" in config.config:
                model = "horse"
            print(
                "Weights not found!\nRun : ./download-weights.sh StyleGAN2-%s" % (model))
            sys.exit(1)
        self.G = stylegan2.models.load(os.path.join(config.weights, "G.pth"))
        self.D = stylegan2.models.load(os.path.join(config.weights, "D.pth"))

    def has_discriminator(self):
        return True

    def generate(self, z, minibatch=None):
        if minibatch is None:
            return self.G(z)
        else:
            assert z.shape[0] % minibatch == 0
            gen_images = []
            for i in range(0, z.shape[0] // minibatch):
                z_minibatch = z[i*minibatch:(i+1)*minibatch, :]
                gen_images.append(self.G(z_minibatch))
            gen_images = torch.cat(gen_images)
            return gen_images

    def discriminate(self, images, minibatch=None):
        if minibatch is None:
            return self.D(images)
        else:
            assert images.shape[0] % minibatch == 0
            discriminations = []
            for i in range(0, images.shape[0] // minibatch):
                images_minibatch = images[i*minibatch:(i+1)*minibatch, :]
                discriminations.append(self.D(images_minibatch))
            discriminations = torch.cat(discriminations)
            return discriminations
