import time

import retro
import numpy as np
import cv2
import neat
import pickle
import gym
from movie_agent import MovieAgent

from neat import visualize as vis

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz\bin'


class Agent(object):
    def __init__(self, genome, config, visualize=False):
        self.genome = genome
        self.config = config
        self.visualize = visualize

    def run(self):
        self.env = retro.make('Airstriker-Genesis', state=retro.State.DEFAULT, scenario=None)

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        imgarray = []
        start_time = time.time()

        while not done:
            if self.visualize:
                self.env.render()

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)

            # actions = [actions[0], 0.0, 0.0, 0.0, 0.0, 0.0, actions[1], actions[2], 0.0, 0.0, 0.0, 0.0]

            ob, rew, done, info = self.env.step(actions)

            fitness += rew
            if time.time() - start_time >= 45:
                break

        # print(fitness)
        self.env.reset()
        return fitness


def eval_genomes(genome, config):
    agent = Agent(genome, config)
    return agent.run()


if __name__ == '__main__':
    for iteration in range(1, 10000, 1000):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config-feedforward.txt')

        p = neat.Population(config)
        restored = False
        if iteration != 1:
            while not restored:
                try:
                    p = neat.Checkpointer.restore_checkpoint('checkpoints12/neat-checkpoint-' + str(iteration))
                    restored = True
                except:
                    iteration -= 1

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(10, filename_prefix='./checkpoints12/neat-checkpoint-'))

        pe = neat.ParallelEvaluator(10, eval_genomes)

        winner = p.run(pe.evaluate, 1000)

        with open('winner12.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

        agent = MovieAgent(winner, config, iteration=iteration+100)
        agent.run()

        vis.draw_net(config, winner, True)
        vis.plot_stats(stats, ylog=False, view=True)
        vis.plot_species(stats, view=True)
