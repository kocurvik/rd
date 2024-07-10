import json
import os

import numpy as np
from matplotlib import pyplot as plt

from utils.data import experiments, iterations_list, colors

large_size = 20
small_size = 16


def draw_results_pose_auc_10(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment])

    plt.xlim([5.0, 1.9e4])
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')
    else:
        plt.legend()
        plt.show()


def draw_results_k_med(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = [np.abs(r['k1'] - r['k1_gt']) for r in iter_results]
            errs.extend([np.abs(r['k2'] - r['k2_gt']) for r in iter_results])
            errs = np.array(errs)
            errs[np.isnan(errs)] = 1.0
            med = np.median(errs)

            xs.append(mean_runtime)
            ys.append(med)

        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment])

    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('Median absolute $\\lambda$ error', fontsize=large_size)
    plt.ylim([0.0, 0.3])
    plt.xlim([5.0, 1.9e4])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.savefig(f'figs/{title}_k.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved k: {title}')
    else:
        plt.legend()
        plt.show()


def draw_graphs(name):
    with open(os.path.join('results', f'{name}.json'), 'r') as f:
        results = json.load(f)
    draw_results_pose_auc_10(results, experiments, iterations_list, title=name)
    draw_results_k_med(results, experiments, iterations_list, title=name)




if __name__ == '__main__':
    draw_graphs('st_vitus_all-graph-pairs-features_superpoint_noresize_2048-LG_eq')
    draw_graphs('st_vitus_all-graph-pairs-features_superpoint_noresize_2048-LG')
    draw_graphs('rotunda_new-graph-pairs-features_superpoint_noresize_2048-LG_eq')
    draw_graphs('rotunda_new-graph-pairs-features_superpoint_noresize_2048-LG')