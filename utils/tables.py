import json
import os

import numpy as np

from utils.data import basenames_pt, basenames_eth

eq_order = ['Fns_7pt', 'Feq_7pt', 'Feq_7pt_s3', 'kFk_8pt', 'kFk_9pt', 'F_7pt', 'F_7pt_s3', 'k2k1_9pt', 'k2Fk1_10pt']
neq_order = ['Fns_7pt', 'F_7pt', 'F_7pt_s3', 'k2k1_9pt', 'k2Fk1_10pt']


incdec = [1, 1, -1, -1, -1, 1, 1, 1]

def table_text(dataset_name, eq_rows, neq_rows, sarg):
    leq = len(eq_rows)
    lneq = len(neq_rows)

    rd_val = '0'
    if sarg == 3:
        rd_val = '-0.9'
    if sarg < 3:
        rd_vals = '0.0, -0.6, -1.2'
    elif sarg == 3:
        rd_vals ='-0.6, -0.9, -1.2'
    if sarg == 2:
        comment = '%'
        leq -= 2
        lneq -= 1
    else:
        comment = ''



    table_f_string = (
        f'\\begin{{tabular}}{{ c | r c c | c c c c c | c c | c}}\n'
        f'    \\toprule\n'
        f'    & & & & \\multicolumn{{7}}{{c}}{{Poselib - {dataset_name}}} \\\\\n'
        f'    \\midrule\n'
        f'    & Minimal & Refinement & Sample & AVG $(^\\circ)$ $\\downarrow$ & MED $(^\\circ)$ $\\downarrow$ & AUC@5 $\\uparrow$ & @10 & @20 & AVG $\\epsilon(\\lambda)$ $\\downarrow$ & MED $\\epsilon(\\lambda)$ $\\downarrow$ & Time (ms) $\\downarrow$ \\\\\n'
        f'    \\midrule\n'
        f'    \\multirow{{{leq}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{$\\lambda_1 = \\lambda_2$}}}} '
        f'    & 7pt \\F & \\F & {rd_val} & {eq_rows[0]} \\\\\n'
        f'    & 7pt \\F & \\Fk & {rd_val} & {eq_rows[1]} \\\\\n'
        f'    {comment}& 7pt \\F & \\Fk & $\\{{{rd_vals}\\}}$ & {eq_rows[2]} \\\\\n'
        f'    & 8pt \\Fk & \\Fk & \\ding{{55}} & {eq_rows[3]} \\\\\n'
        f'    & 9pt \\Fk & \\Fk & \\ding{{55}} & {eq_rows[4]} \\\\\n'
        f'    & 7pt \\F & \\Fkk & 0 & {eq_rows[5]} \\\\\n'
        f'    {comment}& 7pt \\F & \\Fkk & $\\{{-1.2, -0.6, 0\\}}$ & {eq_rows[6]} \\\\\n'
        f'    %\\cmidrule{{2-12}}\n'
        f'    & 9pt \\Fkk & \\Fkk & \\ding{{55}} & {eq_rows[7]} \\\\\n'
        f'    & 10pt \\Fkk & \\Fkk & \\ding{{55}} & {eq_rows[8]} \\\\ \n'
        f'    \\midrule\n'
        f'    \\midrule\n'
        f'    \\multirow{{{lneq}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{$\\lambda_1 \\neq \\lambda_2$}}}} '
        f'    & 7pt \\F & \\F & {rd_val} & {neq_rows[0]} \\\\\n'
        f'    & 7pt \\F & \\Fkk & {rd_val} & {neq_rows[1]} \\\\\n'
        f'    {comment}& 7pt \\F & \\Fkk & $\\{{{rd_vals}\\}}$ & {neq_rows[2]} \\\\\n'
        f'    %\\cmidrule{{2-12}}\n'
        f'    & 9pt \\Fkk & \\Fkk & \\ding{{55}} & {neq_rows[3]} \\\\\n'
        f'    & 10pt \\Fkk & \\Fkk & \\ding{{55}} & {neq_rows[4]} \\\\ \n'
        f'    \\bottomrule\n'
        f'\\end{{tabular}}'
    )
    return table_f_string




def get_rows(results, order):
    num_rows = []

    for experiment in order:
        exp_results = [x for x in results if x['experiment'] == experiment]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])
        p_auc_5 = np.mean(p_res[:5])
        p_auc_10 = np.mean(p_res[:10])
        p_auc_20 = np.mean(p_res)
        p_avg = np.mean(p_errs)
        p_med = np.median(p_errs)

        k_errs = np.array([0.5 * (np.abs(r['k1'] - r['k1_gt']) + np.abs(r['k2'] - r['k2_gt'])) for r in exp_results])
        # k_errs.extend([np.abs(r['k2'] - r['k2_gt']) for r in exp_results])
        # k_errs = np.array(k_errs)
        k_errs[np.isnan(k_errs)] = 4.0
        k_avg = np.mean(k_errs)
        k_med = np.median(k_errs)

        times = [r['info']['runtime'] for r in exp_results]
        time_avg = np.mean(times)

        num_rows.append([p_avg, p_med, p_auc_5, p_auc_10, p_auc_20, k_avg, k_med, time_avg])

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    lens = np.array([[len(x) for x in y] for y in text_rows])
    arr = np.array(num_rows)
    for j in range(len(text_rows[0])):
        idxs = np.argsort(incdec[j] * arr[:, j])
        text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
        text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    max_len = np.max(lens, axis=0)
    phantoms = max_len - lens
    for i in range(len(text_rows)):
        for j in range(len(text_rows[0])):
            if phantoms[i, j] > 0:
                text_rows[i][j] = '\\phantom{' + (phantoms[i, j] * '1') + '}' + text_rows[i][j]

    return [' & '.join(row) for row in text_rows]

def generate_table(dataset, i, feat):
    if dataset == 'pt':
        basenames = basenames_pt
        name = '\\Phototourism'
    elif dataset == 'eth3d':
        basenames = basenames_eth
        name = '\\ETH'
    elif dataset == 'rotunda':
        basenames = ['rotunda_new']
        name = '\\ROTUNDA'
    elif dataset == 'vitus':
        basenames = ['st_vitus_all']
        name = '\\VITUS'

    else:
        raise ValueError

    if i > 0:
        name = name + f' - Synth {"XABC"[i]}'

    if i > 0:
        neq_results_type = f'pairs-features_{feat}_noresize_2048-LG-synth{i}'
        eq_results_type =  f'pairs-features_{feat}_noresize_2048-LG-syntheq{i}'
    else:
        neq_results_type = f'pairs-features_{feat}_noresize_2048-LG'
        eq_results_type = f'pairs-features_{feat}_noresize_2048-LG_eq'
    # results_type = 'graph-SIFT_triplet_correspondences'

    neq_results = []
    eq_results = []
    for basename in basenames:
        json_path = os.path.join('results', f'{basename}-{neq_results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            neq_results.extend(json.load(f))
        json_path = os.path.join('results', f'{basename}-{eq_results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            eq_results.extend(json.load(f))

    print("Data loaded")

    print(30 * '*')
    print(30 * '*')
    print(30 * '*')
    print("Printing: ", name)
    print(30 * '*')

    neq_rows = get_rows(neq_results, neq_order)
    eq_rows = get_rows(eq_results, eq_order)
    print(table_text(name, eq_rows, neq_rows, i))

if __name__ == '__main__':
    for features in ['superpoint', 'sift']:
        generate_table('rotunda', 0, features)
        generate_table('vitus', 0, features)

    for i in range(1, 4):
        generate_table('pt', i, 'superpoint')
    for i in range(1, 4):
        generate_table('eth3d', i, 'superpoint')
