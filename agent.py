import time

import retro
import numpy as np
import cv2
import neat
import pickle
import gym
from movie_agent import MovieAgent


class Agent(object):
    def __init__(self, genome, config, visualize=False):
        self.genome = genome
        self.config = config
        self.visualize = visualize

    def work(self):
        self.env = retro.make('Airstriker-Genesis')

        self.env.reset()

        ob, _, _, done = self.env.step(self.env.action_space.sample())

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

            actions = [actions[0], 0.0, 0.0, 0.0, 0.0, 0.0, actions[1], actions[2], 0.0, 0.0, 0.0, 0.0]

            ob, rew, done, info = self.env.step(actions)
            # print(done)

            fitness += rew
            if time.time() - start_time >= 60:
                break

        # print(fitness)
        self.env.reset()
        return fitness


def eval_genomes(genome, config):
    worky = Agent(genome, config)
    return worky.work()


if __name__ == '__main__':
    for iteration in range(1, 10000, 100):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config-feedforward.txt')

        p = neat.Population(config)
        if iteration != 1:
            p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-' + str(iteration-2))
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(10, filename_prefix='./checkpoints/neat-checkpoint-'))

        pe = neat.ParallelEvaluator(10, eval_genomes)

        winner = p.run(pe.evaluate, 101)

        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

        # print('\nBest genome:\n{!s}'.format(winner))
        #
        # print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        worky = MovieAgent(winner, config, iteration=iteration+100)
        worky.work()

        # visualize.draw_net(config, winner, True)
        # visualize.plot_stats(stats, ylog=False, view=True)
        # visualize.plot_species(stats, view=True)
