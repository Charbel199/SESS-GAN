from logger.log import LoggerService
from typing import List
import random
import numpy as np
import torch
logger = LoggerService.get_instance()


class Agent:
    fitness = 0

    def __init__(self, network):
        self.network = network

    def mutate(self):
        pass


class GeneticAlgorithm:
    def __int__(self, network, remaining_population_percentage=0.1, population_size=50):
        self.population_size = population_size
        self.remaining_population_percentage = remaining_population_percentage
        self.network = network
        pass

    def generate_agents(self) -> List[Agent]:
        return [Agent(self.network) for _ in range(self.population_size)]

    def execute(self, generations, threshold):

        agents = self.generate_agents()
        agents = self.mutation(agents)

        for i in range(generations):
            logger.info(f'Generation {str(i)}:')
            agents = self.fitness(agents)
            agents = self.selection(agents)
            # If statisfactory
            # Keep selected agents

            # If not satisfactory
            agents = self.cross_over(agents)
            agents = self.mutation(agents)

            if any(agent.fitness < threshold for agent in agents):
                logger.info(f'Threshold met at generation {str(i)} !')

            if i % 100:
                pass

        return agents[0]

    def selection(self, agents: List[Agent]) -> List[Agent]:
        agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
        agents = agents[:int(self.remaining_population_percentage * len(agents))]
        return agents

    def mutation(self, agents: List[Agent]) -> List[Agent]:
        for agent in agents:
            if random.uniform(0.0, 1.0) <= 0.1:
                weights = agent.network.weights
                shapes = [a.shape for a in weights]
                flattened = np.concatenate([a.flatten() for a in weights])
                randint = random.randint(0, len(flattened) - 1)
                flattened[randint] = np.random.randn()
                newarray = []
                indeweights = 0
                for shape in shapes:
                    size = np.product(shape)
                    newarray.append(flattened[indeweights: indeweights + size].reshape(shape))
                    indeweights += size
                agent.network.weights = newarray
        return agents

    def fitness(self, agents: List[Agent]) -> List[Agent]:
        for agent in agents:
            # Run simulation

            # Compute score

            # Update fitness score of agent
            agent.fitness = 1
        return agents

    def cross_over(self, agents: List[Agent]):
        offspring = []
        with torch.no_grad():
            for _ in range((self.population_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(self.network)
                child2 = Agent(self.network)

                shapes = [a.shape for a in parent1.network.weights]

                genes1 = np.concatenate([a.flatten() for a in parent1.network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.network.weights])

                split = random.randint(0, len(genes1) - 1)
                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())

                child1.network.weights = self.unflatten(child1_genes, shapes)
                child2.network.weights = self.unflatten(child2_genes, shapes)

                offspring.append(child1)
                offspring.append(child2)
        agents.extend(offspring)
        return agents

    def unflatten(self, flattened, shapes):
        newarray = []
        index = 0
        for shape in shapes:
            size = np.product(shape)
            newarray.append(flattened[index: index + size].reshape(shape))
            index += size
        return newarray
