from simulated_annealing import *
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

def sim_annealing_wrapper(file, stop_score, stop_time, seed):
	graph = read_nxgraph(file)
	init_temp = 2.5
	num_steps = 100000000	#Just set really high so it doesnt quit at a specific step
	start_time = time.time()
	data = simulated_annealing(init_temp, num_steps, graph, seed, stop_time=stop_time, do_pbar=False)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['seed'] = seed
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['stop_time'] = stop_time
		results[file]['gamma'] = (stop_score-data[0])/stop_score

def parrallel_sim_annealing_wrapper(file, stop_score, stop_time, seed):
	graph = read_nxgraph(file)
	init_temp = 2.5
	num_steps = 100000000 	#Just set really high so it doesnt quit at a specific step
	start_time = time.time()
	data = simulated_annealing_tensor(num_steps, graph, init_temp, seed, stop_time=stop_time, do_pbar=False)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['seed'] = seed
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['stop_time'] = stop_time 
		results[file]['gamma'] = (stop_score-data[0])/stop_score


#itterate through graphs
threads = []
stop_time = 3600
seed = 0
for i in range(len(graph_id)):
	file = f'data/syn/powerlaw_{graph_nodes[i]}_ID{graph_id[i]}.txt'
	new_thread = threading.Thread(target=sim_annealing_wrapper, args=(file, float(L2A_scores[i]), stop_time, seed))
	threads.append(new_thread)
	new_thread.start()
	print('started new thread', i)

#join all threads
for thread in threads:
	thread.join()

#Save results
filename = 'simulated_annealing_results2.json'
pretty_filename = 'simulated_annealing_results_pretty2.json'
f = open(filename, 'w')
json.dump(results, f, indent = 4)

pretty_results = results
for graph in results.keys():
	results[graph].pop('solution')
	results[graph].pop('best_scores')
	results[graph].pop('scores')

f = open(pretty_filename, 'w')
json.dump(pretty_results, f, indent = 4)


#find all higher scores (or matching)
for graph in results.keys():
	if results[graph]['score'] >= results[graph]['stop_score']:
		print(f'"{graph}" beat or matched L2A, gamma: {results[graph]["gamma"]}')

print('done')