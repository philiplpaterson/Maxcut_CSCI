import sys, math

import copy
import time
from typing import List, Union
import numpy as np
import random
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut

from itertools import combinations


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

	#High difference in edge number over two classesß
	# for i in range(len(node_degrees)//2):
	# 	init_guess[node_degrees[i][0]] = 1

	#Evenly distributed edge number over two classes
	for i in range(len(node_degrees)):
		if i%2==0:
			init_guess[node_degrees[i][0]] = 1

	return init_guess

def simulated_annealing(init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
	print('simulated_annealing')
	
	adj_matrix = nx.to_numpy_array(graph)
	num_nodes = graph.number_of_nodes()

	init_solution = np.concatenate((np.zeros(num_nodes // 2, dtype=int), np.ones(num_nodes // 2, dtype=int)))

	start_time = time.time()
	curr_solution = copy.deepcopy(init_solution)
	curr_score = obj_maxcut(curr_solution, adj_matrix)
	init_score = curr_score

	best_solution = curr_solution
	best_score = curr_score

	for k in range(num_steps):
		# The temperature decreases
		temperature = init_temperature * (1 - (k + 1) / num_steps)

		new_solution = curr_solution.copy()
		idx = np.random.randint(0, num_nodes)
		new_solution[idx] = (new_solution[idx] + 1) % 2
		
		new_score = obj_maxcut(new_solution, adj_matrix)

		delta_e = curr_score - new_score
		if delta_e < 0:
			curr_solution = new_solution
			curr_score = new_score
		else:
			prob = np.exp(- delta_e / (temperature + 1e-6))
			if prob > random.random():
				curr_solution = new_solution
				curr_score = new_score

		if new_score>best_score:
			best_score = new_score
			best_solution = new_solution

	print("score, init_score of simulated_annealing", best_score, init_score)
	running_duration = time.time() - start_time
	print('running_duration: ', running_duration)
	return best_score, best_solution

if __name__ == '__main__':


	# run alg
	# init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))

	# read data
	graph = read_nxgraph('data/syn/powerlaw_500_ID0.txt')
	init_temperature = 1.25
	num_steps = 10000
	sa_score, sa_solution = simulated_annealing(init_temperature, num_steps, graph)

	print('Gamma:', (1470-sa_score)/1470)






