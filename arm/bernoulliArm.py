import random


class BernoulliArm(object):
    def __init__(self, p):
        self.p = p

    def draw(self):
        # reward system based on Bernoulli
        if random.random() > self.p:
            return 0.0

        return 1.0
