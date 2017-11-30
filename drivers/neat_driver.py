from pytocl.driver import Driver
from pytocl.car import Command
import neat
from copy import deepcopy

class NEATDriver(Driver):

    ticks = 0
    distance_raced = 0
    ticks_off_track = 0
    ticks_collision = 0
    position = 0
    prev_state = None

    def __init__(self, network=None, handler=None, logdata=False):
        self.handler = handler
        self.network = network
        super(NEATDriver, self).__init__(logdata=logdata)

    def set_handler(self, handler):
        self.handler = handler

    def set_network(self, network):
        self.network = network

    def reward(self, final_state):
        # Parameters directly copied from Cardamone et al.
        C = 8000
        alpha = 5
        beta = 10
        gamma = 50 # increased from 3

        # Factor for how many cars the agent has overtaken
        started = self.position
        finished = final_state.race_position
        delta = finished - started

        off_track_penalty = alpha * self.ticks_off_track
        collision_penalty = beta * self.ticks_collision

        return C - off_track_penalty - collision_penalty - gamma * delta + self.distance_raced

    def gather_statistics(self, carstate):
        if self.ticks == 0:
            self.position = carstate.race_position

        # Count number of ticks
        self.ticks += 1
        self.distance_raced = carstate.distance_raced

        # count number of ticks spent off the track
        off_track = not (-1 < carstate.distance_from_center < 1)
        self.ticks_off_track += 1 if off_track else 0

        # count number of ticks spent colliding with opponents
        collision = any([distance < 5 for distance in carstate.opponents])
        self.ticks_collision += 1 if collision else 0

    def drive(self, carstate):

        # Save interesting statistics for reward function
        self.gather_statistics(carstate)

        if self.ticks == 1500:
            reward = self.reward(carstate)
            self.handler.stop_evaluation(reward)

        # if self.ticks % 120 == 0:
            # print("Ticks:", self.ticks)
            # print("Collisions:", self.ticks_collision)
            # print("Distance:", self.distance_raced)
            # print("Reward:", self.reward(carstate))

        prev_state = deepcopy(carstate)

        # If ANY sensor notices an opponent, we engage overtaking behaviour
        overtaking_situation = any([distance != 200.0 for distance in carstate.opponents])

        if overtaking_situation and self.network:
            # [0::3] means take each 3rd index, starting with 0
            inp = list(carstate.distances_from_edge[0::3])
            inp.append(carstate.speed_x)

            actuators = self.network.activate(inp)

            command = Command()

            command.steering = self.steering_ctrl.control(
                actuators[0],
                carstate.current_lap_time
            )

            if actuators[1] > 0:
                command.accelerator = 1

                if carstate.rpm > 8000:
                    command.gear = carstate.gear + 1

            else:
                command.brake = 1

            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

            if not command.gear:
                command.gear = carstate.gear or 1

            return command

        else:
            # Then just do default behaviour
            return super(NEATDriver, self).drive(carstate)
