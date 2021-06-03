import argparse
import os
import torch
import numpy as np
import pickle

# pymoo stuff, stays here
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.visualization.scatter import Scatter


# these ones to change
from config import get_config
from problem import GenerationProblem
from operators import get_operators


# global iteration


def runner(config, iteration):

    # global iteration
    # global config

    print('start')

    def save_callback(algorithm, iteration):
        # global iteration
        # global config

        iteration += 1
        if iteration % config.save_each == 0 or iteration == config.generations:

            sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
            X = np.stack([p.X for p in sortedpop])

            ls = config.latent(config)
            ls.set_from_population(X)

            with torch.no_grad():
                generated = algorithm.problem.generator.generate(
                    ls, minibatch=config.batch_size)
                name = f"genetic-it-{iteration}.jpg" if iteration < config.generations else "genetic-it-final.jpg"
                algorithm.problem.generator.save(
                    generated, os.path.join(config.tmp_folder, name))

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

    res = minimize(
        problem,
        algorithm,
        ("n_gen", config.generations),
        save_history=True,
        verbose=True,
    )

    pickle.dump(dict(
        X=res.X,
        F=res.F,
        G=res.G,
        CV=res.CV,
    ), open(os.path.join(config.tmp_folder, "genetic_result"), "wb"))

    # again
    sortedpop = sorted(res.pop, key=lambda p: p.F)
    X = np.stack([p.X for p in sortedpop])

    ls = config.latent(config)
    ls.set_from_population(X)

    torch.save(ls.state_dict(), os.path.join(
        config.tmp_folder, "ls_result"))

    X = np.atleast_2d(res.X)

    ls.set_from_population(X)

    with torch.no_grad():
        generated = problem.generator.generate(ls)

    problem.generator.save(generated, os.path.join(
        config.tmp_folder, "output.jpg"))
