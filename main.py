import pyglet
import pymunk
import pymunk.pyglet_util
import itertools
import operator
import math
import neat


Position = tuple[float, float]


WINDOW = pyglet.window.Window(800, 600)
DRAW_OPTIONS = pymunk.pyglet_util.DrawOptions()


class Car(pymunk.Body):
    def __init__(self, start: Position, brain: neat.nn.feed_forward.FeedForwardNetwork, genome: neat.genome.DefaultGenome):
        super().__init__()

        self.brain = brain
        self.genome = genome

        self.position = start
        self.shape = pymunk.Circle(self, 30)
        self.shape.density = 1
        self.shape.elasticity = 0
        self.shape.collision_type = 5

        self.previous_position = start

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

        # Permanent force forwards
        # self.impulse = lambda dt: self.apply_force_at_local_point((10 ** 7 * dt, 0))
        self.impulse = lambda dt: self.apply_force_at_local_point((10 ** 7 * dt / 2, 0))
        pyglet.clock.schedule_interval(self.impulse, 1/120)

        self.thought = lambda dt: self.think(dt)
        pyglet.clock.schedule_interval(self.thought, 1/120)

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

    def think(self, dt: float) -> None:
        output = self.brain.activate(self.get_distances())
        i = output.index(max(output))  # Get node of highest value from outputs

        if i == 0:
            self.angle += 0.11
            self.space.reindex_shapes_for_body(self)
        elif i == 1:
            self.angle -= 0.11
            self.space.reindex_shapes_for_body(self)

    def reward_movement(self) -> None:
        self.distance_traveled += math.dist(self.previous_position, self.position)

    def kill_if_stuck(self) -> None:
        d = math.dist(self.previous_position, self.position)

        if d < 3:
            self.die()
        else:
            self.previous_position = self.position

    def die(self) -> bool:
        pyglet.clock.unschedule(self.impulse)
        pyglet.clock.unschedule(self.thought)

        self.genome.fitness = self.distance_traveled  # FIXME: alguma outra funcao que nao seja f(x)
        print(self.distance_traveled)

        self.space.remove(self, self.shape, *self.sensors)

        return True


class Sensor(pymunk.Segment):
    def __init__(self, car: Car, offset: Position, distance: float):
        super().__init__(car, (0, 0), offset, 1)

        self.sensor = True
        self.color = (255, 0, 0, 0.5)
        self.distance = distance


class Environment(pymunk.Space):
    def __init__(self):
        super().__init__()

        self.create_walls((50, 50), [(500, 0), (200, 200), (0, 100), (-200, 200), (-500, 0), (0, -500)])
        self.create_walls((50, 150), [(400, 0), (100, 100), (0, 100), (-100, 100), (-300, 0), (0, -300)])

        self.damping = 0.5

        coll_handler = self.add_wildcard_collision_handler(9)
        coll_handler.pre_solve = lambda a, s, d: a.shapes[1].body.update_distance(a, a.shapes[1].collision_type)
        coll_handler.separate = lambda a, s, d: a.shapes[1].body.not_sensing(a.shapes[1].collision_type)

        car_crash_handler = self.add_collision_handler(5, 9)
        car_crash_handler.pre_solve = lambda a, s, d: a.shapes[0].body.die()

        car_coll_handler = self.add_collision_handler(5, 5)
        car_coll_handler.pre_solve = lambda a, s, d: False

        pyglet.clock.schedule_interval(self.step, 1/120)

    def create_walls(self, start: Position, moves: list[Position]) -> None:
        tuple_add = lambda a, b: tuple(map(operator.add, a, b))

        points = itertools.accumulate(moves, initial=start, func=tuple_add)

        for a, b in itertools.pairwise(points):
            wall = pymunk.Segment(self.static_body, a, b, 1)
            wall.elasticity = 0
            wall.collision_type = 9

            self.add(wall)

    def run_simulation(self, genomes, config) -> None:
        self.running = 1

        for id, g in genomes:
            nn = neat.nn.FeedForwardNetwork.create(g, config)
            g.fitness = 0

            car = Car((100, 100), nn, g)
            car.add_to_space(self)

        exit_if_empty = lambda dt: self.close_gracefully() if len(self.bodies) == 0 else None
        pyglet.clock.schedule_interval(exit_if_empty, 1)

        reward_if_moving = lambda dt: [car.reward_movement() for car in self.bodies]
        pyglet.clock.schedule_interval(reward_if_moving, 1/5)

        kill_if_not_moving = lambda dt: [car.kill_if_stuck() for car in self.bodies]
        pyglet.clock.schedule_interval(kill_if_not_moving, 1)

        pyglet.app.run()

        if self.running == 1:
            raise Exception('App closed prematurely')

        pyglet.clock.unschedule(exit_if_empty)
        pyglet.clock.unschedule(reward_if_moving)
        pyglet.clock.unschedule(kill_if_not_moving)

    def close_gracefully(self):
        pyglet.app.exit()
        self.running = 0


ENVIRONMENT = Environment()


@WINDOW.event
def on_draw() -> None:
    pyglet.gl.glClearColor(0, 0, 0, 0)
    WINDOW.clear()
    ENVIRONMENT.debug_draw(DRAW_OPTIONS)


if __name__ == '__main__':
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        p.run(ENVIRONMENT.run_simulation)
    except:
        print("Interrupted!")
