import numpy as np
import torch
from scipy.stats import truncnorm

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.model.sampling import Sampling

from pymoo.algorithms.so_de import DE
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling


class TruncatedNormalRandomSampling(Sampling):
    def __init__(self, var_type=np.float):
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return truncnorm.rvs(-2, 2, size=(n_samples, problem.n_var)).astype(np.float32)


class NormalRandomSampling(Sampling):
    def __init__(self, mu=0, std=1, var_type=np.float):
        super().__init__()
        self.mu = mu
        self.std = std
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return np.random.normal(self.mu, self.std, size=(n_samples, problem.n_var))


class BinaryRandomSampling(Sampling):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < self.prob).astype(np.bool)


def get_operators(config):
    if "DeepMindBigGAN256" in config.config or config.config == "DeepMindBigGAN512":
        mask = ["real"]*config.dim_z + ["bool"]*config.num_classes

        # real_sampling = None
        # if config.config == "DeepMindBigGAN256" or config.config == "DeepMindBigGAN512":
        real_sampling = TruncatedNormalRandomSampling()

        sampling = MixedVariableSampling(mask, {
            "real": real_sampling,
            "bool": BinaryRandomSampling(prob=5/1000)
        })

        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "bool": get_crossover("bin_hux", prob=0.2)
        })

        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm", prob=0.5, eta=3.0),
            "bool": get_mutation("bin_bitflip", prob=10/1000)
        })

        return dict(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

    elif config.config.split("_")[0] == "StyleGAN2":
        return dict(
            sampling=NormalRandomSampling(),
            crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
            mutation=get_mutation("real_pm", prob=0.5, eta=3.0)
        )

    else:
        raise Exception("Unknown config")


# def save_callback(algorithm):
#     global iteration
#     global config

#     iteration += 1
#     if iteration % config.save_each == 0 or iteration == config.generations:
#         sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
#         X = np.stack([p.X for p in sortedpop])

#         ls = config.latent(config)
#         ls.set_from_population(X)

#         with torch.no_grad():
#             generated = algorithm.problem.generator.generate(
#                 ls, minibatch=config.batch_size)
#             name = "genetic-it-%d.jpg" % (
#                 iteration) if iteration < config.generations else "genetic-it-final.jpg"
#             algorithm.problem.generator.save(
#                 generated, os.path.join(config.tmp_folder, name))
#             display(Image(os.path.join(config.tmp_folder, name)))


# class AlgorithmSetup():

#     def __init__(self, config, model_name, algorithm_name):

#         self.algo = None
#         self.config = config
#         self.model_name = model_name
#         self.algorithm_name = algorithm_name

#     def set_algorithm(self):

#         if model_name == 'stylegan':
#             if algorithm_name == 'de':
#                 algorithm = DE(
#                     pop_size=100,
#                     sampling=LatinHypercubeSampling(
#                         iterations=100, criterion="maxmin"),
#                     variant="DE/rand/1/bin",
#                     CR=0.5,
#                     F=0.3,
#                     dither="vector",
#                     jitter=False)
#                 else:
#                     algorithm = get_algorithm(
#                         'ga',
#                         pop_size=config.pop_size,
#                         sampling=self.algo_config["sampling"],
#                         crossover=self.algo_config["crossover"],
#                         mutation=self.algo_config["mutation"],
#                         eliminate_duplicates=True,
#                         callback=save_callback,
#                         **(self.config.algorithm_args[self.config.algorithm] if "algorithm_args" in self.config and self.config.algorithm in self.config.algorithm_args else dict()))

#                 self.algo = algorithm

#         def set_settings_ga(self):

#             if self.model_name == 'biggan':

#                 mask = ["real"]*config.dim_z + ["bool"]*config.num_classes

#                 real_sampling = TruncatedNormalRandomSampling()

#                 sampling = MixedVariableSampling(mask, {
#                     "real": real_sampling,
#                     "bool": BinaryRandomSampling(prob=5/1000)
#                 })

#                 crossover = MixedVariableCrossover(mask, {
#                     "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
#                     "bool": get_crossover("bin_hux", prob=0.2)
#                 })

#                 mutation = MixedVariableMutation(mask, {
#                     "real": get_mutation("real_pm", prob=0.5, eta=3.0),
#                     "bool": get_mutation("bin_bitflip", prob=10/1000)
#                 })

#                 self.algo_config = dict(
#                     sampling=sampling,
#                     crossover=crossover,
#                     mutation=mutation)

#             elif self.model_name == 'stylegan':
#                 self.algo_config = dict(
#                     sampling=NormalRandomSampling(),
#                     crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
#                     mutation=get_mutation("real_pm", prob=0.5, eta=3.0))

#             return

#         def set_settings_de(self):

#             # placeholder

#             if self.model_name == 'biggan':

#                 mask = ["real"]*config.dim_z + ["bool"]*config.num_classes

#                 # real_sampling = None
#                 # if config.config == "DeepMindBigGAN256" or config.config == "DeepMindBigGAN512":
#                 real_sampling = TruncatedNormalRandomSampling()

#                 sampling = MixedVariableSampling(mask, {
#                     "real": real_sampling,
#                     "bool": BinaryRandomSampling(prob=5/1000)
#                 })

#                 crossover = MixedVariableCrossover(mask, {
#                     "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
#                     "bool": get_crossover("bin_hux", prob=0.2)
#                 })

#                 mutation = MixedVariableMutation(mask, {
#                     "real": get_mutation("real_pm", prob=0.5, eta=3.0),
#                     "bool": get_mutation("bin_bitflip", prob=10/1000)
#                 })

#                 self.algo_config = dict(
#                     sampling=sampling,
#                     crossover=crossover,
#                     mutation=mutation)

#             elif self.model_name == 'stylegan':
#                 self.algo_config = dict(
#                     sampling=NormalRandomSampling(),
#                     crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
#                     mutation=get_mutation("real_pm", prob=0.5, eta=3.0))

#             return


class AlgorithmSetup2():

    def __init__(self, config, model_name, algorithm_name):

        self.algo = None
        self.config = config
        self.model_name = model_name
        self.algorithm_name = algorithm_name

        def set_algo(self):
            iteration = 0

            def save_callback(algorithm):
                global iteration
                global config

                iteration += 1
                if iteration % config.save_each == 0 or iteration == config.generations:
                    sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
                    X = np.stack([p.X for p in sortedpop])

                    ls = config.latent(config)
                    ls.set_from_population(X)

                    with torch.no_grad():
                        generated = algorithm.problem.generator.generate(
                            ls, minibatch=config.batch_size)
                        name = "genetic-it-%d.jpg" % (
                            iteration) if iteration < config.generations else "genetic-it-final.jpg"
                        algorithm.problem.generator.save(
                            generated, os.path.join(config.tmp_folder, name))
                        display(Image(os.path.join(config.tmp_folder, name)))

            problem = GenerationProblem(config)
            operators = get_operators(config)

            if not os.path.exists(config.tmp_folder):
                os.mkdir(config.tmp_folder)

            algorithm = get_algorithm(
                config.algorithm,
                pop_size=config.pop_size,
                sampling=operators["sampling"],
                crossover=operators["crossover"],
                mutation=operators["mutation"],
                eliminate_duplicates=True,
                callback=save_callback,
                **(config.algorithm_args[config.algorithm] if "algorithm_args" in config and config.algorithm in config.algorithm_args else dict())
            )

            return algorithm
