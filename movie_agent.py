import time

import retro
import numpy as np
import cv2
import neat
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class MovieAgent(object):
    def __init__(self, genome, config, visualize=True, iteration=300):
        self.genome = genome
        self.config = config
        self.visualize = visualize
        self.iteration = str(iteration)

    def run(self):
        self.env = retro.make('Airstriker-Genesis', state=retro.State.DEFAULT, scenario=None)

        self.env.reset()
        video_recorder = None

        video_recorder = VideoRecorder(self.env, './video/airstrike12_' + self.iteration + '.mp4', enabled=True)

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
                video_recorder.capture_frame()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)

            # actions = [actions[0], 0.0, 0.0, 0.0, 0.0, 0.0, actions[1], actions[2], 0.0, 0.0, 0.0, 0.0]

            ob, rew, done, info = self.env.step(actions)
            # print(done)

            fitness += rew
            if time.time() - start_time >= 300:
                break

        video_recorder.close()

        video_recorder.enabled = False

        self.env.reset()
        self.env.close()
        # print(fitness)
        return fitness