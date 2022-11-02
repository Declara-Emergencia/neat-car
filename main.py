import math
import itertools
import functools
import operator
import concurrent.futures
import sys
import pymunk
import neat
from neat.graphs import feed_forward_layers
import pygame
import pymunk.pygame_util


Position = tuple[float, float]


class Car(pymunk.Body):
    def __init__(self, start: Position, brain: neat.nn.feed_forward.FeedForwardNetwork, genome: neat.genome.DefaultGenome):
        super().__init__()

        self.alive = True

        self.brain = brain
        self.genome = genome

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
        output = self.brain.activate(self.get_distances())
        i = output.index(max(output))  # Get node of highest value from outputs

        if i == 0:
            self.angle += 0.11
            self.space.reindex_shapes_for_body(self)
        elif i == 1:
            self.angle -= 0.11
            self.space.reindex_shapes_for_body(self)

    def reward_movement(self) -> None:
        self.distance_traveled += math.dist(self.previous_position_movement, self.position)

        self.previous_position_movement = self.position


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

        self.genome.fitness = self.distance_traveled

        self.space.remove(self, self.shape, *self.sensors)

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


def evaluate_genome(genome: neat.DefaultGenome, config: neat.Config) -> float:
    nn = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = 0
    car = Car((100, 100), nn, genome)

    frames = 1
    env = Environment()
    car.add_to_space(env)

    while car.alive:
        car.think()
        car.accelerate()
        env.step(1/120)

        if frames % 44 == 0: # 5 times every "second"
            car.reward_movement()

        if frames % 120 == 0: # every "second"
            car.kill_if_stuck()

        if frames > 50000:
            print('Took too long')
            car.die()
            return car.genome.fitness

        frames += 1

    #print(car.genome.fitness)

    return car.genome.fitness


def simulate_genome(genome: neat.DefaultGenome, config: neat.Config) -> None:
    nn = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car((100, 100), nn, genome)

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


class CustomReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, pop, species, best_genome):
        print(best_genome)
        connections = [cg.key for cg in best_genome.connections.values() if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        print(layers)

        if best_genome.fitness > 16000:
            try:
                simulate_genome(best_genome, config)
            except Exception as e:
                print('Finished:', e)
                sys.exit(0)

                return

#    def start_generation(self, generation):
#        if generation > 10:
#            sys.exit(0)


if __name__ == '__main__':
    # Set configuration file
    config_path = "./config-feedforward.txt"
    #config_path = "./config-noat.txt" # Without augmenting topologies
    #config_path = "./config-hiddennodes.txt" # With 5 hidden nodes at the start
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(CustomReporter())

    try:
        pe = neat.ParallelEvaluator(4, evaluate_genome)
        p.run(pe.evaluate)
    except Exception as e:
        print("Interrupted!", e)
