import random, torch, tqdm
import numpy as np

import networkx as nx
from util import read_nxgraph
from util import obj_maxcut_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print('Device set to', device)

def calculate_population_diversity(population):
    num_individuals = population.shape[0]
    population_reshaped = population.view(num_individuals, -1).float()  # Reshape to treat each individual as a vector
    hamming_distances = torch.cdist(population_reshaped, population_reshaped, p=0)  # Calculate Hamming distances
    hamming_distances_sum = torch.sum(hamming_distances)  # Sum of all Hamming distances
    # Exclude diagonal elements (distances of individuals with themselves)
    population_diversity = (hamming_distances_sum - torch.trace(hamming_distances)) / (num_individuals * (num_individuals - 1))
    return population_diversity

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
	population = (torch.rand((population_size, num_nodes), device=device)<0.5).to(torch.int8)
	for k in range(num_generations):

		# Update temperature/mutation rate
		temp = init_temp * (1 - (k) / num_generations)

		# Get fitness scores/Maxcut scores
		fitnesses = obj_maxcut_batch(population, adj_matrix)
		gen_best = torch.max(fitnesses).item()
		
		# Sort population by fittness scores/Maxcut scores
		sorted_indices = torch.argsort(fitnesses, descending=True)
		population = population[sorted_indices]
		if gen_best > best_score:
			best_score = gen_best
			best_solution = population[0].to("cpu").numpy()

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
			#pbar.set_description(f'Score: {best_score}, temp: {temp:.3f}, div: {calculate_population_diversity(population):.3f}')
			pbar.set_description(f'Score: {best_score}, temp: {temp:.3f}')
			pbar.update()

	return best_solution, best_score

from util import obj_maxcut
import copy
def simulated_annealing(init_guess:np.ndarray, init_temperature: int, num_steps: int, graph: nx.Graph, seed:int, do_pbar=True):

	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	adj_matrix = nx.to_numpy_array(graph)
	num_nodes = graph.number_of_nodes()

	#init_solution = np.concatenate((np.zeros(num_nodes // 2, dtype=int), np.ones(num_nodes // 2, dtype=int)))
	init_solution = init_guess

	curr_solution = copy.deepcopy(init_solution)
	curr_score = obj_maxcut(curr_solution, adj_matrix)

	best_solution = curr_solution
	best_score = curr_score

	if do_pbar:
		pbar = tqdm.tqdm(range(num_steps), f'Simulated Annealing, Score: {best_score}')
	ctr = 0
	for k in range(num_steps):
		# The temperature decreases
		temperature = init_temperature * (1 - (k + 1) / num_steps)

		new_solution = curr_solution.copy()
		new_solution[k%num_nodes] = (new_solution[k%num_nodes] + 1) % 2
		ctr+=1

		new_score = obj_maxcut(new_solution, adj_matrix)
		if new_score>best_score:
			best_score = new_score
			best_solution = new_solution

		delta_e = curr_score - new_score
		if delta_e < 0 or ctr>=num_nodes:
			curr_solution = new_solution
			curr_score = new_score
			ctr = 0
		else:
			prob = np.exp(- delta_e / (temperature + 1e-6))
			if prob > random.random():
				curr_solution = new_solution
				curr_score = new_score
				ctr = 0
		
		if do_pbar:
			pbar.set_description(f'Simulated Annealing, Score: {best_score}, {curr_score}')
			pbar.update()

	if do_pbar:
		pbar.close()

	#print("score, init_score of simulated_annealing", best_score, init_score)
	return best_score, best_solution


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
 
	genetic_params = {
		'num_generations': 2000,
		'population_size': 2000, 
		'init_temp': 0.05,
		'graph': read_nxgraph('data/syn/powerlaw_500_ID1.txt'), 
		'seed': 0, 
		'do_pbar': True
	}

	print(f'temp: {genetic_params["init_temp"]}, seed: {genetic_params["seed"]}, population size: {genetic_params["population_size"]}')
	best_sol, score = genetic_vectorized(**genetic_params)

	simulated_annealing_params = {
		'init_guess': best_sol,
		'init_temperature': 1.5,
		'num_steps': 100000,
		'graph': genetic_params["graph"],
		'seed': 0,
		'do_pbar': True
	}

	best_sol, score = simulated_annealing(**simulated_annealing_params)


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