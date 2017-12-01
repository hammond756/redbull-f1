import neat
from drivers.neat_driver import NEATDriver
from pytocl.driver import Driver
from pytocl.protocol import Client
import sys, os
import subprocess
import functools
from multiprocessing import Pool, Array
import time

class Evaluation():

    _reward = None
    client = None
    process = None

    def __init__(self, driver, port=3002):
        self.driver = driver
        self.port = port

    def evaluate(self):

        # in headless mode, torcs won't close easily, so this makes sure
        # the old processes are killed
        os.system('pkill torcs')

        # Start TORCS
        local_dir = os.path.dirname(__file__)
        config_dir = os.path.join(local_dir, 'torcs_configuration')
        xml_path = os.path.join(config_dir, 'evaluation_{}.xml'.format(self.port))
        xml_path = os.path.abspath(xml_path)

        self.process = subprocess.Popen('torcs -r {} -nofuel -nolaptime &'.format(xml_path), shell=True)

        # Make driver report to self (to monitor ticks and collect reward)
        self.driver.set_handler(self)

        # Initialze server
        self.client = Client(driver=self.driver, port=self.port)
        self.client.run()

        # Wait for reward to be returned
        while not self._reward or self._reward is 0:
            pass

        return self._reward

    # This method is called by the driver when a certain number of ticks is
    # reached
    def stop_evaluation(self, reward):
        self._reward = reward
        self.client.stop()
        os.system('pkill torcs')

    def get_reward(self):
        return self._reward

def eval_single_genome(genome, config):
    print(list(available_ports))

    try:
        idx_port = list(available_ports).index(1)
    except:
        idx_port = 0

    available_ports[idx_port] = 0
    port = list(range(3001, 3011))[idx_port]

    network = neat.nn.RecurrentNetwork.create(genome, config)
    evalulation_server = Evaluation(NEATDriver(network=network), port=port)
    genome.fitness = evalulation_server.evaluate()

    available_ports[idx_port] = 1

    return


# multi specifies multiprocessing or not. But it doens't work, so stick with
# False
def eval_genomes(genomes, config, multi=False):
    os.system('pkill torcs')
    if multi:
        raise NotImplementedError()
    else:
        for gen_id, genome in genomes:
            eval_single_genome(genome, config)
            print("Fitness", genome.fitness)

    os.system('pkill torcs')

if __name__ == '__main__':

    available_ports = Array('i', 10*[1])

    torcs_server_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    winner = population.run(eval_genomes, 10)

    os.system('torcs -nofuel -nolaptime &')

    winning_network = neat.nn.RecurrentNetwork.create(winner, config)
    driver = NEATDriver(network=winning_network)
    client = Client(driver=driver)
    client.run()
