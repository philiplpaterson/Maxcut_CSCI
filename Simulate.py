from simulated_annealing import *
from genetic_algorithm import *
import threading, csv, json
import numpy as np
from util import read_nxgraph


file = open(r'result\result_syn_L2A.csv')
csv_reader = csv.reader(file)

data = []
for row in csv_reader:
	data.append(row)

L2A_scores = np.array(data)[61:211, -22]
graph_nodes = np.array(data)[61:211, 1]
graph_id = np.array(data)[61:211, 2]

results_lock = threading.Lock()
results = dict()

def sim_annealing_wrapper(file, init_temp, num_steps, stop_score, seed, do_pbar):
	graph = read_nxgraph(file)
	start_time = time.time()
	data = simulated_annealing(init_temp, num_steps, graph, seed, do_pbar=do_pbar)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['max_iterations'] = num_steps
		results[file]['seed'] = seed
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['gamma'] = (stop_score-data[0])/stop_score
		results[file]['init_temp'] = init_temp


def parrallel_sim_annealing_wrapper(file, init_temp, num_steps, stop_score, seed, do_pbar):
	graph = read_nxgraph(file)
	start_time = time.time()
	data = simulated_annealing_tensor(init_temp, num_steps, graph, seed, do_pbar=do_pbar)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['max_iterations'] = num_steps
		results[file]['seed'] = seed
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['gamma'] = (stop_score-data[0])/stop_score
		results[file]['init_temp'] = init_temp

def sim_genetic_wrapper(file, init_mutation, init_temp, num_generations, population_size, num_steps, seed, stop_score, do_pbar):
	start_time = time.time()

	genetic_params = {
		'num_generations': num_generations,
		'population_size': population_size, 
		'init_temp': init_mutation,
		'graph': read_nxgraph(file), 
		'seed': seed, 
		'do_pbar': do_pbar
	}
	best_sol_genetic, score_genetic = genetic_vectorized(**genetic_params)

	simulated_annealing_params = {
		'init_guess': best_sol_genetic,
		'init_temperature': init_temp,
		'num_steps': num_steps,
		'graph': genetic_params["graph"],
		'seed': seed,
		'do_pbar': do_pbar
	}

	score_sa, best_sol_sa = simulated_annealing_gen(**simulated_annealing_params)

	results[file] = dict()
	results[file]['genetic_score'] = score_genetic
	results[file]['sim_annealing_score'] = score_sa
	results[file]['solution'] = best_sol_sa.tolist()
	results[file]['num_iterations'] = num_steps
	results[file]['num_generations'] = num_generations
	results[file]['population_size'] = population_size
	results[file]['seed'] = seed
	results[file]['stop_score'] = stop_score
	results[file]['run_time'] = time.time()-start_time
	results[file]['gamma'] = (stop_score-score_sa)/stop_score
	results[file]['init_temp'] = init_temp
	results[file]['init_mutation'] = init_mutation


	#Save results
	filename = 'genetic.json'
	f = open(filename, 'w')
	json.dump(results, f, indent = 4)
	f.close()


# #parameters
# seed = 0
# num_steps = 1000 #Max seems to be around 250k in 1hr
# init_temp = 2.5

# #itterate through graphs
# threads = []
# for i in range(len(graph_id)):
# 	file = f'data/syn/powerlaw_{graph_nodes[i]}_ID{graph_id[i]}.txt'
# 	new_thread = threading.Thread(target=sim_annealing_wrapper, args=(file, init_temp, num_steps, float(L2A_scores[i]), seed, i==len(graph_id)-1))
# 	threads.append(new_thread)
# 	new_thread.start()
# 	print('started new thread', i)

# #join all threads
# for thread in threads:
# 	thread.join()

# #Save results
# filename = 'test.json'
# pretty_filename = 'test_pretty3.json'
# f = open(filename, 'w')
# json.dump(results, f, indent = 4)
# f.close()

# pretty_results = results
# for graph in results.keys():
# 	results[graph].pop('solution')
# 	results[graph].pop('best_scores')
# 	results[graph].pop('scores')

# f = open(pretty_filename, 'w')
# json.dump(pretty_results, f, indent = 4)
# f.close()

# print('done')

# '''
# first test did not set seed for each attempt, only set seed once globally in SA
# temp = 2.5
# num_iter = 100,000
# '''

seed = 0
init_mutation= 0.05
init_temp = 1.5
num_generations = 2000
population_size = 2000
num_steps = 100000
do_pbar = True

for i in range(len(graph_id)):
	file = f'data/syn/powerlaw_{graph_nodes[i]}_ID{graph_id[i]}.txt'
	print(file)
	stop_score = float(L2A_scores[i])
	sim_genetic_wrapper(file, init_mutation, init_temp, num_generations, population_size, num_steps, seed, stop_score, do_pbar)
	print()
	torch.cuda.empty_cache()

pretty_filename = 'genetic_pretty.json'
pretty_results = results
for graph in results.keys():
	results[graph].pop('solution')
f = open(pretty_filename, 'w')
json.dump(pretty_results, f, indent = 4)
f.close()

#find all higher scores (or matching)
for graph in results.keys():
	if results[graph]['sim_annealing_score'] >= results[graph]['stop_score']:
		print(f'"{graph}" beat or matched L2A, gamma: {results[graph]["gamma"]}')