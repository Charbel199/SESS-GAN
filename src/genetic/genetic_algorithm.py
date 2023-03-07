from logger.log import LoggerService
from typing import List
import random
import numpy as np
import torch
from evaluate.generate_samples import generate_map
from conf import ModelConfig
import requests
import json
import os
from helpers.save import torch_save
import re

logger = LoggerService.get_instance()


def get_most_recent_generation_dir(directory):
    # Create a regular expression to match the numeric part of directory names
    pattern = re.compile(r"\d+$")

    # Initialize the maximum numeric value to zero and the corresponding directory name to None
    max_value = 0
    max_directory = None

    # Loop through the directories in the directory
    for subdir in os.listdir(directory):
        # Check if the subdirectory is a directory
        if os.path.isdir(os.path.join(directory, subdir)):
            # Extract the numeric part of the directory name
            match = pattern.search(subdir)
            if match:
                value = int(match.group(0))
                # Update the maximum numeric value and directory name if this directory has a bigger value
                if value > max_value:
                    max_value = value
                    max_directory = subdir

    # Print the maximum directory name
    return max_directory, max_value


def average(lst):
    return sum(lst) / len(lst)


def matrix2d_to_string(matrix):
    matrix_string = ""
    for row in matrix:
        for i in row:
            matrix_string += f"{str(i)} "
        matrix_string += "\n"
    return matrix_string


def flatten_model(model):
    """
    Example output:
    {
    0:
        {
            "head.conv.weight":[.....],
            "head.conv.bias":[......],
            ...
        }
    1:
        {
        ...
        }

    }
    """
    model_dict = {}
    for scale, generator in enumerate(model):
        model_dict[scale] = {}
        keys = list(generator.state_dict().keys())
        weight_strings = [s for s in keys if '.weight' in s or '.bias' in s]
        for weight_string in weight_strings:
            weights = generator.state_dict()[weight_string]
            flattened = weights.flatten()
            model_dict[scale][weight_string] = flattened
    return model_dict


def unflatten_model(model, model_dict):
    """
    Updates the model with the model_dict values
    """
    for scale, generator in enumerate(model):
        keys = list(generator.state_dict().keys())
        weight_strings = [s for s in keys if '.weight' in s or '.bias' in s]
        for weight_string in weight_strings:
            weights = generator.state_dict()[weight_string]
            shape = weights.shape
            generator.state_dict()[weight_string] = model_dict[scale][weight_string].reshape(shape)

    return model


class Agent:
    def __init__(self, network):
        self.network = network
        self.fitness = 0

    def mutate(self, mutation_rate=0.01):

        model_dict = flatten_model(self.network)
        for scale in model_dict.keys():
            weight_strings = list(model_dict[scale].keys())
            for weight_string in weight_strings:
                weights = model_dict[scale][weight_string]
                model_dict[scale][weight_string] = weights * (1 + mutation_rate * torch.randn_like(weights))
        self.network = unflatten_model(self.network, model_dict)


class GeneticAlgorithm:
    def __init__(self, network, remaining_population_percentage=0.1, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.remaining_population_percentage = remaining_population_percentage
        self.network = network
        self.mutation_rate = mutation_rate

    def generate_agents(self) -> List[Agent]:
        return [Agent(self.network) for _ in range(self.population_size)]

    def execute(self, generations, threshold):
        logger.info(f"Generating agents")
        agents = self.generate_agents()
        logger.info(f"Applying initial mutations")
        agents = self.mutation(agents)

        for i in range(generations):
            logger.info(f'Generation #{str(i)}:')
            agents = self.fitness(agents)
            agents = self.selection(agents)
            logger.info(f'Best fitness {agents[-1].fitness}')
            # If statisfactory
            # Keep selected agents
            if any(agent.fitness > threshold for agent in agents):
                logger.info(f'Threshold met at generation {str(i)} !')

            # If not satisfactory
            logger.info(f"Applying crossover")
            agents = self.cross_over(agents)
            logger.info(f"Applying mutation")
            agents = self.mutation(agents)

        return agents[0]

    def selection(self, agents: List[Agent]) -> List[Agent]:
        agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
        agents = agents[:int(self.remaining_population_percentage * len(agents))]
        return agents

    def mutation(self, agents: List[Agent]) -> List[Agent]:
        for agent in agents:
            agent.mutate(mutation_rate=self.mutation_rate)
        return agents

    def fitness(self, agents: List[Agent]) -> List[Agent]:
        for agent in agents:
            # Run simulation

            # Compute score

            # Update fitness score of agent
            agent.fitness = random.uniform(0.5, 0.96)
        return agents

    def cross_over(self, agents: List[Agent]):
        offspring = []
        with torch.no_grad():
            for _ in range((self.population_size - len(agents)) // 2):
                # Randomly choose 2 parents
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)


                model_dict1 = flatten_model(parent1.network)
                model_dict2 = flatten_model(parent2.network)
                child_model_dict1={}
                child_model_dict2={}
                for scale in model_dict1.keys():
                    weight_strings = list(model_dict1[scale].keys())

                    child_model_dict1[scale]={}
                    child_model_dict2[scale]={}

                    for weight_string in weight_strings:
                        weights1 = model_dict1[scale][weight_string]
                        weights2 = model_dict2[scale][weight_string]
                        # Apply crossover
                        split = random.randint(0, len(weights1) - 1)
                        child1_genes = torch.cat((weights1[:split], weights2[split:]), dim=0)
                        child2_genes = torch.cat((weights1[split:], weights2[:split]), dim=0)
                        # Populating model_dicts for children
                        child_model_dict1[scale][weight_string] = child1_genes
                        child_model_dict2[scale][weight_string] = child2_genes

                # Initialize children with random weights (parent weights in this case)
                child1 = Agent(parent1.network)
                child2 = Agent(parent2.network)
                # Update children weights
                child1.network = unflatten_model(child1.network, child_model_dict1)
                child2.network = unflatten_model(child2.network, child_model_dict2)

                # Append children
                offspring.append(child1)
                offspring.append(child2)

        logger.info(f"Agents len is {len(agents)} and offspring len {len(offspring)}")
        agents.extend(offspring)
        logger.info(f"New Agents len is {len(agents)}")
        return agents

    def unflatten(self, flattened, shapes):
        newarray = []
        index = 0
        for shape in shapes:
            size = np.product(shape)
            newarray.append(flattened[index: index + size].reshape(shape))
            index += size
        return newarray
