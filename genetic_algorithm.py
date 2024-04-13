import random, torch, tqdm
import numpy as np

import networkx as nx
from util import read_nxgraph
from util import obj_maxcut_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to', device)

def genetic_vectorized(num_generations: int, population_size:int, init_temp: int, graph: nx.Graph, seed: int, do_pbar=True):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	temp = init_temp
	best_score = 0
	survivor_percent = 0.10
	best_solution = None

	adj_matrix = torch.tensor(nx.to_numpy_array(graph), device=device)
	num_nodes = graph.number_of_nodes()

	if do_pbar:
		pbar = tqdm.tqdm(range(num_generations), f'Simulated Annealing, Score: {best_score}')

	#generate initial population
	population = (torch.rand((population_size, num_nodes), device=device)<0.5).int()
	for k in range(num_generations):

		# Update temperature/mutation rate
		temp = init_temp * (1 - (k) / num_generations)

		# Get fitness scores/Maxcut scores
		fitnesses = obj_maxcut_batch(population, adj_matrix)
		gen_best = torch.max(fitnesses).item()
		if gen_best > best_score:
			best_score = gen_best
		
		# Sort population by fittness scores/Maxcut scores
		sorted_indices = torch.argsort(fitnesses, descending=True)
		population = population[sorted_indices]

		#save top 10%
		next_gen = torch.zeros_like(population)
		num_survive = int(population_size*survivor_percent)
		next_gen[:num_survive] = population[:num_survive]

		# Generate offspring from the top 50%
		parent_indices = torch.randint(0, population_size // 2, ((population_size - num_survive)*2,)) 	#generate vector of length (population_size - num_survive)*2, of random parents from the top 50% (*2 bc u need 2 parents)
		parent1 = population[parent_indices[:parent_indices.shape[0]//2]] 								#parent 1 is from first half of random top 50%
		parent2 = population[parent_indices[parent_indices.shape[0]//2:]] 								#parent 2 is from second half of random top 50%
		crossover_points = torch.randint(0, 2, (population_size - num_survive, num_nodes), device=device)
		offspring = parent1 * crossover_points + parent2 * (1 - crossover_points) 						#1 for parent1 gene, and 0 for parent 2 gene

		# Apply mutations
		mutations = (torch.rand(offspring.shape[0], num_nodes, device=device) < temp)
		offspring ^= mutations

		# Set next generation
		next_gen[num_survive:] = offspring
		population = next_gen

		if do_pbar:
			pbar.set_description(f'Genetic Algorithm, Score: {best_score}, {gen_best}, {temp:.3f}')
			pbar.update()



if __name__ == '__main__':

	'''
	NOTE: population size doesn't change ideal starting temp,
	for 1000 generations it seems like 0.05 is the ideal starting temp, but number of itterations might change that
 
	NOTE: after futher testing it seems like an initial temperature of 0.05 is ideal for any number of itterations. 
  	it might not be for lower population sizes, but with larger population sizes it seems to converge to 0.05
	'''

	# params = {
	# 	'num_generations': 2000,
	# 	'population_size': 1000, 
	# 	'init_temp': 0.05, 
	# 	'graph': read_nxgraph('data/syn/powerlaw_500_ID0.txt'), 
	# 	'seed': 0, 
	# 	'do_pbar': True
	# } #= 1460 = 0.088 gamma
 
	params = {
		'num_generations': 10000,
		'population_size': 1000, 
		'init_temp': 0.05, 
		'graph': read_nxgraph('data/syn/powerlaw_500_ID0.txt'), 
		'seed': 0, 
		'do_pbar': True
	}

	print(f'temp: {params["init_temp"]}, seed: {params["seed"]}, population size: {params["population_size"]}')
	genetic_vectorized(**params)

	# Test all 500 node graphs
	# for i in range(30):
	# 	params = {
	# 		'num_generations': 2000,
	# 		'population_size': 1000, 
	# 		'init_temp': 0.05, 
	# 		'graph': read_nxgraph(f'data/syn/powerlaw_500_ID{i}.txt'), 
	# 		'seed': 0, 
	# 		'do_pbar': True
	# 	}
	# 	print(f'500 node graph ID{i}')
	# 	genetic_vectorized(**params)