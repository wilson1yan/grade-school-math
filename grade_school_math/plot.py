import numpy as np
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--idx', type=int, default=0)
parser.add_argument('--answer_file', type=str, default='answers_test.json')
parser.add_argument('--matrix_file', type=str, default='ensemble_matrices_test.npy')
args = parser.parse_args()

# 11 27 55

print(f'Plotting idx {args.idx}')
answers = json.load(open(args.answer_file, 'r'))
n_models, n_data = len(answers), len(answers[0])
is_correct = np.zeros((n_models, n_data), dtype=bool)
for i in range(n_models):
    for j in range(n_data):
        is_correct[i, j] = answers[i][j][1]
print(np.nonzero(is_correct[0]))

matrices = np.load(args.matrix_file)

answers = [answers[i][args.idx] for i in range(n_models)]
matrix = matrices[args.idx]

is_correct = np.array([a[1] for a in answers])
is_correct = is_correct[:, None] & is_correct[None]
is_correct = 1 - is_correct.astype(float)
is_correct[is_correct == 0] = 0.25
is_correct[is_correct == 1] = 0.75
plt.imshow(is_correct.T, aspect='auto', cmap='bwr', vmin=0, vmax=1)
for (i, j), value in np.ndenumerate(matrix):
    plt.text(i, j, f'%.3f' % value, va='center', ha='center')
plt.xlabel('sample from ith model')
plt.ylabel('likelihood for jth model')
plt.xticks(np.arange(n_models))
plt.yticks(np.arange(n_models))
plt.savefig(f'figs/fig_{args.idx}.png')
