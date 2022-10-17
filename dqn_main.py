# https://github.com/keon/deep-q-learning/blob/master/dqn.py

import math
import itertools
import functools
import operator
import concurrent.futures
import pymunk
import pygame
import pymunk.pygame_util

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


Position = tuple[float, float]


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Car(pymunk.Body):
    def __init__(self, start: Position, agent: DQNAgent):
        super().__init__()

        self.alive = True

        self.brain = agent

        self.position = start
        self.shape = pymunk.Circle(self, 30)
        self.shape.density = 1
        self.shape.elasticity = 0
        self.shape.collision_type = 5

        self.previous_position_stuck = start
        self.previous_position_movement = start

        self.distance_traveled = 0

        self.sensor_distance = 200

        self.sensors = []

        for i in range(5):
            angle = -math.pi/2 + math.pi/4 * i

            sensor = Sensor(self, (math.cos(angle) * self.sensor_distance,
                                   math.sin(angle) * self.sensor_distance),
                            self.sensor_distance)
            sensor.color = (255/5 * i, 0, 255, 1)
            sensor.collision_type = i

            self.sensors.append(sensor)

    def update_distance(self, arbiter: pymunk.Arbiter, i: int) -> bool:
        if i >= 5:
            return True

        contact_point = arbiter.contact_point_set.points[0].point_a

        self.sensors[i].distance = math.dist(self.position, contact_point)

        return False

    def not_sensing(self, i: int) -> None:
        if i < 5:
            self.sensors[i].distance = self.sensor_distance

    def get_distances(self) -> list[float]:
        return list(map(lambda d: d / self.sensor_distance, [sensor.distance for sensor in self.sensors]))

    def add_to_space(self, space: pymunk.Space) -> None:
        space.add(self, self.shape, *self.sensors)

    def accelerate(self) -> None:
        self.apply_force_at_local_point((10 ** 6 / 10, 0))

    def think(self) -> None:
        state = self.get_distances()
        i = self.brain.act(np.reshape(state, [1, 5]))

        if i == 0:
            self.angle += 0.11
            self.space.reindex_shapes_for_body(self)
        elif i == 1:
            self.angle -= 0.11
            self.space.reindex_shapes_for_body(self)

        agent.memorize(np.reshape(state, [1, 5]), i, self.reward_movement(), np.reshape(self.get_distances(), [1, 5]), not self.alive)

    def reward_movement(self) -> float:
        reward = math.dist(self.previous_position_movement, self.position)
        self.distance_traveled += reward

        self.previous_position_movement = self.position

        return reward


    def milestone(self, milestone: pymunk.Shape) -> bool:
        self.distance_traveled += 1000

        self.space.remove(milestone)

        return False

    def kill_if_stuck(self) -> None:
        d = math.dist(self.previous_position_stuck, self.position)

        if d < 3:
            self.die()
        else:
            self.previous_position_stuck = self.position

    def die(self) -> bool:
        self.alive = False

        self.space.remove(self, self.shape, *self.sensors)

        print(self.distance_traveled)

        return True


class Sensor(pymunk.Segment):
    def __init__(self, car: Car, offset: Position, distance: float):
        super().__init__(car, (0, 0), offset, 1)

        self.sensor = True
        self.color = (255, 0, 0, 0.5)
        self.distance = distance


class Milestone(pymunk.Circle):
    def __init__(self, position: Position, space: pymunk.Space):
        super().__init__(space.static_body, 10, position)

        self.sensor = True
        self.collision_type = 7
        self.color = (255, 255, 0, 0.5)

        space.add(self)


class Environment(pymunk.Space):
    def __init__(self):
        super().__init__()

        self.create_walls((50, 50), [(500, 0), (200, 200), (0, 100), (-200, 200), (-500, 0), (0, -500)])
        self.create_walls((50, 150), [(400, 0), (100, 100), (0, 100), (-100, 100), (-300, 0), (0, -300)])

        self.create_milestones([(200, 100), (300, 100), (400, 100), (500, 100),
                                (550, 150), (600, 200), (650, 250),
                                (650, 350), (600, 400), (550, 450),
                                (500, 500), (400, 500), (300, 500), (200, 500),
                                (100, 500), (100, 400), (100, 300), (100, 200)])

        self.damping = 0.5

        coll_handler = self.add_wildcard_collision_handler(9)
        coll_handler.pre_solve = lambda a, s, d: a.shapes[1].body.update_distance(a, a.shapes[1].collision_type)
        coll_handler.separate = lambda a, s, d: a.shapes[1].body.not_sensing(a.shapes[1].collision_type)

        car_crash_handler = self.add_collision_handler(5, 9)
        car_crash_handler.pre_solve = lambda a, s, d: a.shapes[0].body.die()

        milestone_handler = self.add_collision_handler(5, 7)
        milestone_handler.pre_solve = lambda a, s, d: a.shapes[0].body.milestone(a.shapes[1])

        car_coll_handler = self.add_collision_handler(5, 5)
        car_coll_handler.pre_solve = lambda a, s, d: False

    def create_walls(self, start: Position, moves: list[Position]) -> None:
        tuple_add = lambda a, b: tuple(map(operator.add, a, b))

        points = itertools.accumulate(moves, initial=start, func=tuple_add)

        for a, b in itertools.pairwise(points):
            wall = pymunk.Segment(self.static_body, a, b, 1)
            wall.elasticity = 0
            wall.collision_type = 9

            self.add(wall)

    def create_milestones(self, positions: list[Position]) -> None:
        for p in positions:
            milestone = Milestone(p, self)


def simulate_agent(agent: DQNAgent) -> None:
    car = Car((100, 100), agent)

    pygame.init()
    window = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()

    env = Environment()
    car.add_to_space(env)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    def exit_gracefully() -> bool:
        pygame.display.quit()
        pygame.quit()

        return False

    end_simul = env.add_collision_handler(5, 9)
    end_simul.pre_solve = lambda a, s, d: exit_gracefully()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_gracefully()
                return

        car.think()
        car.accelerate()
        env.step(1/120)

        window.fill((255,255,255))
        env.debug_draw(draw_options)
        pygame.display.flip()

        clock.tick(120)


if __name__ == '__main__':
    batch_size = 32

    agent = DQNAgent(5, 3)

    for e in range(10):
        print('EPISODE:', e)
        car = Car((100, 100), agent)
        env = Environment()
        car.add_to_space(env)
        frames = 1

        while car.alive:
            car.think()

            car.accelerate()
            env.step(1/120)

            if frames % 120 == 0: # every "second"
                car.kill_if_stuck()

            if frames > 50000:
                print('Took too long')
                car.die()
                break

            frames += 1

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    simulate_genome(agent)
