import sys, math, tqdm, torch
import matplotlib.pyplot as plt

import copy
import time
from typing import List, Union
import numpy as np
import random
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut
from util import obj_maxcut2
from util import obj_maxcut_batch

from itertools import combinations

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Solution():
	def __init__(self, solution, score) -> None:
		self.sol = solution
		self.flips = 1
		self.itter = combinations(range(len(solution)), self.flips)
		self.counter = math.comb(len(solution), self.flips)

	def get_next(self):
		"""
		Gets next solution "closest" to the curent solution, such that the closest has the lowest
		number of swaped values, while still being unique so far

		NOTE: upon some simple tests, it seems like this could be much more efficient bye first exploring the 
		neighbors of the highest direct neighbor first - after further testing, for largre graphs, this never
		reaches the second fliping level, so this would have no effect
		"""
		
		#check for next level (need to flip more bits)
		if self.counter == 0:
			self.flips+=1
			self.itter = combinations(range(len(self.sol)), self.flips)
			self.counter = math.comb(len(self.sol), self.flips)

			if self.flips>len(self.sol):
				return None

		#Get next indicies to flip
		self.counter-=1
		indices = next(self.itter)

		#generate next solution array
		next_sol = self.sol.copy()
		for i in indices:
			next_sol[i] = (self.sol[i] + 1) % 2
		
		return next_sol

def get_init_guess(graph: nx.Graph):
	node_degrees = sorted(graph.degree(), key=lambda x: x[1])
	init_guess = np.zeros(len(node_degrees), dtype=int)

	#Evenly distributed edge number over two classes
	for i in range(len(node_degrees)):
		if i%2==0:
			init_guess[node_degrees[i][0]] = 1

	return init_guess

def simulated_annealing(init_temperature: int, num_steps: int, graph: nx.Graph, seed, stop_score=None, stop_time=None, do_pbar=True) -> (int, Union[List[int], np.array], List[int]):

	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	adj_matrix = nx.to_numpy_array(graph)
	num_nodes = graph.number_of_nodes()

	#init_solution = np.concatenate((np.zeros(num_nodes // 2, dtype=int), np.ones(num_nodes // 2, dtype=int)))
	init_solution = get_init_guess(graph)

	start_time = time.time()
	curr_solution = copy.deepcopy(init_solution)
	curr_score = obj_maxcut(curr_solution, adj_matrix)
	init_score = curr_score

	best_solution = curr_solution
	best_score = curr_score
	
	scores = []
	best_scores = []

	if do_pbar:
		pbar = tqdm.tqdm(range(num_steps), f'Simulated Annealing, Score: {best_score}')
	ctr = 0
	num_iter = 0
	for k in range(num_steps):
		# The temperature decreases
		temperature = init_temperature * (1 - (k + 1) / num_steps)

		new_solution = curr_solution.copy()
		new_solution[k%num_nodes] = (new_solution[k%num_nodes] + 1) % 2
		ctr+=1

		new_score = obj_maxcut(new_solution, adj_matrix)

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
		scores.append(curr_score)

		if new_score>best_score:
			best_score = new_score
			best_solution = new_solution
			best_scores.append([best_score, k])
		
		if do_pbar:
			pbar.set_description(f'Simulated Annealing, Score: {best_score}, {curr_score}')
			pbar.update()
		num_iter+=1

		#check for stop
		if (stop_time!=None and time.time()-start_time > stop_time) or (stop_score!=None and best_score>stop_score):
			break

	if do_pbar:
		pbar.close()

	#print("score, init_score of simulated_annealing", best_score, init_score)
	running_duration = time.time() - start_time
	print('running_duration: ', running_duration)
	return best_score, best_solution, best_scores, scores, num_iter


def flip_one_value_vectorized(tensor):
	n = tensor.shape[0]
	identity = torch.eye(n, dtype=torch.int).to(device)
	flipped_matrices = (identity + tensor.repeat(n, 1)) % 2
	return flipped_matrices

def simulated_annealing_tensor(init_temp:float, num_steps: int, graph: nx.Graph, seed, stop_score=None, stop_time=None, do_pbar=True) -> (int, Union[List[int], np.array], List[int]):

	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	adj_matrix = torch.tensor(nx.to_numpy_array(graph)).to(device)
	num_nodes = graph.number_of_nodes()

	#init_solution = np.concatenate((np.zeros(num_nodes // 2, dtype=int), np.ones(num_nodes // 2, dtype=int)))
	init_solution = torch.tensor(get_init_guess(graph)).to(device)

	start_time = time.time()
	curr_solution = init_solution
	curr_score = obj_maxcut2(curr_solution, adj_matrix)
	init_score = curr_score

	best_solution = curr_solution
	best_score = curr_score

	scores =[]
	best_scores = []

	num_iter = 0
	if do_pbar:
		pbar = tqdm.tqdm(range(num_steps), f'Simulated Annealing, Score: {best_score}')
	for k in range(num_steps):
		new_solutions = flip_one_value_vectorized(curr_solution)

		new_scores = obj_maxcut_batch(new_solutions, adj_matrix)
		best_sol_idx = torch.argmax(new_scores)

		temperature = init_temp * (1 - (k) / num_steps)
		#temperature = init_temp / (1 + .99 * np.log(1+k+1))
		#temperature =  init_temp

		# delta_e = curr_score - new_scores[best_sol_idx]
		# if delta_e < 0:
		# 	curr_solution = new_solutions[best_sol_idx]
		# 	curr_score = new_scores[best_sol_idx]
		# else:
		# 	#normalize score tensor and generate sample
		# 	#probability_distribution = torch.nn.functional.softmax(new_scores, dim=-1)
		# 	#probability_distribution = torch.nn.functional.softmax(new_scores*(k/num_steps)*(1/temperature), dim=-1)
		# 	#probability_distribution = torch.nn.functional.softmax(new_scores*(1/temperature), dim=-1)
		# 	delta_e = -new_scores + curr_score
		# 	probs = delta_e/curr_score + torch.exp(-delta_e / (temperature + 1e-6))
		# 	probability_distribution = (probs)/probs.sum()
		# 	sampled_index = torch.multinomial(probability_distribution, 1).item()

		# 	curr_solution = new_solutions[sampled_index]
		# 	curr_score = new_scores[sampled_index]

		delta_e = curr_score - new_scores[best_sol_idx]
		temp = curr_score
		if delta_e < 0:
			temp = new_scores[best_sol_idx]

		delta_e = -new_scores + temp
		probs = delta_e/curr_score + torch.exp(-delta_e / (temperature+1e-6))
		probs = torch.exp(-delta_e / (temperature+1e-6))
		probs+=1e-6
		probability_distribution = (probs)/probs.sum()
		sampled_index = torch.multinomial(probability_distribution, 1).item()

		curr_solution = new_solutions[sampled_index]
		curr_score = new_scores[sampled_index]

		scores.append(curr_score.to('cpu').item())

		if new_scores[best_sol_idx]>best_score:
			best_score = new_scores[best_sol_idx]
			best_solution = new_solutions[best_sol_idx]
			best_scores.append([best_score.to('cpu').item(), k])

		if do_pbar:
			pbar.set_description(f'Simulated Annealing, Score: {best_score}, {curr_score}, {temperature:.3f}')
			pbar.update()
		num_iter+=1

		#check for stop
		if (stop_time!=None and time.time()-start_time > stop_time) or (stop_score!=None and best_score.to('cpu').item()>stop_score):
			break
	if do_pbar:
		pbar.close()

	#print("score, init_score of simulated_annealing", best_score.to('cpu').numpy(), init_score.to('cpu').numpy())
	running_duration = time.time() - start_time
	print('running_duration: ', running_duration)
	return best_score, best_solution, best_scores, scores, num_iter

if __name__ == '__main__':
	#goal 1475...
	# read data
	graph = read_nxgraph('data/syn/powerlaw_500_ID10.txt')
	init_temperature = 2.5 #Best so far
	num_steps = 100000 #Best so far
	# init_temperature = 1.5
	# num_steps = 10000
	result = simulated_annealing(init_temperature, num_steps, graph, 0, stop_score=1473, stop_time=3600)

	print('Solution:', result[1])
	print('Gamma:', (1473.0-result[0])/1473.0)

	graph = read_nxgraph('data/syn/powerlaw_500_ID10.txt')
	init_temperature = 2.5 #Best so far
	num_steps = 100000 #Best so far
	# init_temperature = 1.5
	# num_steps = 10000
	result = simulated_annealing(init_temperature, num_steps, graph, 0, stop_score=1473, stop_time=3600)

	print('Solution:', result[1])
	print('Gamma:', (1473.0-result[0])/1473.0)

	# import matplotlib.pyplot as plt

	# plt.plot(scores, label='Score')
	# plt.plot(best_scores[:,1],best_scores[:,0], label='Best Score')
	# plt.legend()
	# plt.show()


	# # read data
	# graph = read_nxgraph('data/syn/powerlaw_500_ID0.txt')
	# init_temp = 2.5
	# num_steps = 100000 #Best so far
	# sa_score, sa_solution, best_scores, scores= simulated_annealing_tensor(init_temp, num_steps, graph)

	# print('Solution:', sa_solution.to('cpu').numpy())
	# print('Gamma:', (1470-sa_score.to('cpu').item())/1470)
	# print('Temp:', init_temp)

	# plt.plot(best_scores[:,1],best_scores[:,0], label='Best Score')
	# plt.plot(scores, label='Score')
	# plt.legend()
	# plt.show()