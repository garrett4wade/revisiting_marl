import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ar_tie = np.array([0.0045, 0.0161, 0.6495, 0.3287, 0.0011])
ar_tiebreak1 = np.array([0.0082, 0.9333, 0.0118, 0.0010, 0.0457])
ar_tiebreak2 = np.array([4.8756e-05, 5.1848e-04, 1.3930e-04, 2.3134e-05, 9.9927e-01])

idp1 = np.array([5.7101e-07, 1.3071e-06, 2.3476e-06, 1.0000e+00, 3.9427e-07])
idp2 = np.array([1.8897e-05, 3.7677e-05, 3.3077e-06, 9.9994e-01, 5.0567e-06])

for i, prob in enumerate([ar_tie, ar_tiebreak1, ar_tiebreak2, idp1, idp2]):
    # sns.set_style('white')
    fig = plt.figure(figsize=(6, 2))

    plt.rc('xtick', labelsize=20)
    plt.bar(['idle', 'up', 'down', 'right', 'left'],
            prob,
            color=['darkorange', 'darkred', 'darkblue', 'darkgreen', 'darkgoldenrod'])
    ax = plt.gca()
    ax.set_xticks(['idle', 'up', 'down', 'right', 'left'])
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.2)
    # plt.ylabel('Probability')
    # plt.yticks([])
    # plt.xlabel('Action')
    # ax = sns.heatmap(ar_tie.reshape(1, -1), cmap="YlGnBu", cbar=False, annot=True, fmt='.2f', yticklabels=False, xticklabels=False)
    fig.savefig(f'out{i}.png')
print('done')