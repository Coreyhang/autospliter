import matplotlib.pyplot as plt
from parse_sa_log import parse_sa_log
import numpy as np
import matplotlib.ticker as mtick

if __name__ == '__main__':

    file_name = 'jellyfish_1024_3_0'

    init_avg, init_var, init_balance, temperature, delta_cost, cost, taken, avg, var, balance = parse_sa_log(
        'topology/temp/{:s}.txt'.format(file_name))
    x = np.linspace(0, len(temperature), len(temperature))
    avg_relative = [item / init_avg for item in avg]
    var_relative = [item / init_var for item in var]
    balance_relative = [item / init_balance for item in balance]

    font = {
        'family': 'serif',
        'weight': 'normal',
        'size': 6,
    }
    with plt.style.context(['science']):
        fig, ax = plt.subplots(2, 1)
        # ax.plot(x, temperature, label='Temperature')
        # ax.plot(x, delta_cost, label='delta_cost')
        ax[0].plot(x, cost, label='cost')
        ax[0].legend(prop={'size': 5}, loc=0)
        ax[1].plot(x, avg_relative, label='avg')
        ax[1].plot(x, var_relative, label='var')
        ax[1].plot(x, balance_relative, label='balance')
        ax[1].legend(prop={'size': 5}, loc=0)

        # ax[1].set(xlabel='Iteration', fontdict=font)
        ax[1].set_xlabel('Iteration', fontdict=font)
        ax[0].set_ylabel(ylabel='Cost', fontdict=font)
        ax[1].set_ylabel(ylabel='Cost', fontdict=font)

        ax[0].autoscale(tight=True)
        ax[1].autoscale(tight=True)
        ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax[0].tick_params(labelsize=5)
        ax[1].tick_params(labelsize=5)

        fig.savefig('topology/temp/figures/{:s}.pdf'.format(file_name))
        fig.savefig(
            'topology/temp/figures/{:s}.jpg'.format(file_name), dpi=3000)
