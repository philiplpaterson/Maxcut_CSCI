import random, torch, tqdm, math
import numpy as np

import networkx as nx
from util import read_nxgraph
from util import obj_maxcut_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_genome(length: int):
	return (torch.rand(length) > 0.5).int()

class Individual():
	def __init__(self, genome:np.ndarray, temp:int = 0.10):
		self.genome = genome
		self.fitness = 0

	def offspring(self, parent, temp=0.10):
		child_chromosome = torch.zeros_like(self.genome, dtype=torch.int)
		for i in range(len(self.genome)):
			prob = random.random()
			if prob < (1 - temp) / 2:
				child_chromosome[i] = self.genome[i]
			elif prob < (1 - temp):
				child_chromosome[i] = parent.genome[i]
			else:
				child_chromosome[i] = int(random.random() < 0.5)
		return Individual(genome=child_chromosome)

def genetic_simulated_annealing(num_generations: int, population_size:int, init_temp: int, graph: nx.Graph, seed: int, do_pbar=True):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	temp = init_temp
	best_score = 0
	best_solution = None

	adj_matrix = torch.tensor(nx.to_numpy_array(graph)).to(device)
	num_nodes = graph.number_of_nodes()

	if do_pbar:
		pbar = tqdm.tqdm(range(num_generations), f'Simulated Annealing, Score: {best_score}')

	#generate initial population
	population = [Individual(random_genome(num_nodes)) for _ in range(population_size)]
	
	for k in range(num_generations):
		genomes = torch.stack([individual.genome for individual in population], dim=0)
		#print(genomes)

		temp = init_temp * (1 - (k) / num_generations)
		#temp = init_temp / math.log(2 + k)
		#temp = init_temp / math.log(k + 2)
		#temp = init_temp / math.log(k + 2, num_generations + 2) * math.log(k + 2)
		#temp = 1 - ((k+1)/num_generations)


		fitnesses = obj_maxcut_batch(genomes, adj_matrix)
		gen_best = torch.max(fitnesses).item()
		if gen_best > best_score:
			best_score = gen_best

		for i in range(len(population)):
			population[i].fitness = fitnesses[i]

		population.sort(key=lambda x: x.fitness, reverse=True)

		#save top 10%
		next_gen = []
		survivor_percent = 0.1
		num_survive = int(population_size*survivor_percent)
		next_gen.extend(population[:num_survive])

		#generate offspring from the top 50%
		for _ in range(population_size - num_survive):
			parent1 = random.choice(population[:population_size//2])
			parent2 = random.choice(population[:population_size//2])
			child = parent1.offspring(parent2, temp=temp)
			next_gen.append(child)
  
		population = next_gen

		if do_pbar:
			pbar.set_description(f'Genetic Algorithm, Score: {best_score}, {gen_best}, {temp:.3f}')
			pbar.update()



if __name__ == '__main__':
	# genome_length = 10
	# test = Individual(random_genome(genome_length))
	# test2 = Individual(random_genome(genome_length))
	# print('parent #1:', test.genome)
	# print('parent #2:', test2.genome)
	# offspring = test.offspring(test2)
	# print('offspring:', offspring.genome)

	params = {
		'num_generations': 3000,
		'population_size': 10, 
		'init_temp': 0.10, 
		'graph': read_nxgraph('data/syn/powerlaw_500_ID0.txt'), 
		'seed': 0, 
		'do_pbar': True
	}

	genetic_simulated_annealing(**params)
