import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

algos = list(data.keys())

speeds = []
qualities = []
success_probs = []
iters_list = []

for algo in algos:
    entries = data[algo]
    spd_vals = []
    q_vals = []
    prob_vals = []
    iter_vals = []
    for e in entries:
        t = e.get('total_time', np.nan)
        p = e.get('best_measurement_probability',
                  e.get('best_measurement', {}).get('probability', np.nan))
        spd_vals.append(p / t if t and t > 0 else np.nan)
        q_vals.append(e.get('best_value', np.nan))
        prob_vals.append(p)
        if 'iterations' in e and e['iterations'] is not None:
            iter_vals.append(e['iterations'])
        elif 'generations' in e and e['generations'] is not None:
            iter_vals.append(e['generations'])
        elif 'n_evals' in e and e['n_evals'] is not None:
            iter_vals.append(e['n_evals'])
        elif 'cost_function_evals' in e and e['cost_function_evals'] is not None:
            iter_vals.append(e['cost_function_evals'])
        else:
            iter_vals.append(np.nan)
    speeds.append(np.nanmean(spd_vals))
    qualities.append(np.nanmean(q_vals))
    success_probs.append(np.nanmean(prob_vals))
    iters_list.append(np.nanmean(iter_vals))

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Speed
axes[0].bar(algos, speeds)
axes[0].set_title('Средняя скорость успеха\n(probability / time)')
axes[0].set_ylabel('Speed (1/s)')
axes[0].tick_params(axis='x', rotation=45)

# Quality
axes[1].bar(algos, qualities)
axes[1].set_title('Среднее качество решения\n(best_value)')
axes[1].set_ylabel('Best Value')
axes[1].tick_params(axis='x', rotation=45)

# Success Probability
axes[2].bar(algos, success_probs)
axes[2].set_title('Средняя вероятность успеха')
axes[2].set_ylabel('Probability')
axes[2].tick_params(axis='x', rotation=45)

# Iterations
axes[3].bar(algos, iters_list)
axes[3].set_title('Среднее число итераций/оценок')
axes[3].set_ylabel('Iterations / Evals')
axes[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
