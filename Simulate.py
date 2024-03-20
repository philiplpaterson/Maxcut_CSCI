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

def sim_annealing_wrapper(file, stop_score, stop_time):
	graph = read_nxgraph(file)
	init_temp = 2.5
	num_steps = 100000 #Best so far
	start_time = time.time()
	data = simulated_annealing(init_temp, num_steps, graph, stop_score=stop_score, stop_time=stop_time, do_pbar=False)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['stop_time'] = stop_time

def parrallel_sim_annealing_wrapper(file, stop_score, stop_time):
	graph = read_nxgraph(file)
	init_temp = 2.5
	num_steps = 100000 #Best so far
	start_time = time.time()
	data = simulated_annealing_tensor(num_steps, graph, init_temp, stop_score=stop_score, stop_time=stop_time, do_pbar=False)

	with results_lock:
		results[file] = dict()
		results[file]['score'] = data[0]
		results[file]['solution'] = data[1].tolist()
		results[file]['best_scores'] = data[2]	#NOTE: each entry gets saved as a tuple (new_best_score, iteration_timestamp)
		results[file]['scores'] = data[3]
		results[file]['num_iterations'] = data[4]
		results[file]['stop_score'] = stop_score
		results[file]['run_time'] = time.time()-start_time
		results[file]['stop_time'] = stop_time 


#itterate through graphs
threads = []
stop_time = 3600
for i in range(len(graph_id)):
	file = f'data/syn/powerlaw_{graph_nodes[i]}_ID{graph_id[i]}.txt'
	new_thread = threading.Thread(target=sim_annealing_wrapper, args=(file, float(L2A_scores[i]), stop_time))
	threads.append(new_thread)
	new_thread.start()
	print('started new thread', i)

#join all threads
for thread in threads:
	thread.join()

#Save results
f = open('simulated_annealing_results.json', 'w')
json.dump(results, f, indent = 4)


#find all higher scores (or matching)
for graph in results.keys():
	if results[graph]['score'] >= results[graph]['stop_score']:
		print(f'"{graph}" beat or matched L2A, gamma: {(results[graph]["stop_score"]-results[graph]["score"])/results[graph]["stop_score"]}')

print('done')