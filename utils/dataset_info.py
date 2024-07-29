import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_ks(path):
    f = h5py.File(path)

    return [np.array(v)[2, 2] for k, v in f.items()]


if __name__ == '__main__':
    paths = ['/mnt/d/Research/data/RD/rotunda_new/parameters_rd.h5', '/mnt/d/Research/data/RD/cathedral/parameters_rd.h5',]
    for path in paths:
        ks = get_ks(path)
        print(len(ks))
        print(np.unique(ks))
        print(len(np.unique(ks)))

    large_size = 24
    small_size = 20

    bins = np.linspace(-1.7, 0.1, 7)
    print(bins)
    sns.histplot(ks, bins=bins)
    plt.xlabel('λ', fontsize=large_size)
    plt.ylabel('Count', fontsize=large_size)
    plt.xticks(bins)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    plt.savefig('figs/cathedral_hist.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

