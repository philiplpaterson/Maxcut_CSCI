import numpy as np
import matplotlib.pyplot as plt
import json

file = open("simulated_annealing_results.json")
results = json.load(file)


scores = np.array(results['data/syn/powerlaw_100_ID1.txt']['scores'])
best_scores = np.array(results['data/syn/powerlaw_100_ID1.txt']['best_scores'])

plt.plot(scores, label='Score')
plt.plot(best_scores[:,1], best_scores[:,0], label='Best Score')
plt.legend()
plt.show()