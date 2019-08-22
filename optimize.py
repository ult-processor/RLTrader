import multiprocessing
import numpy as np
import pickle
from multiprocessing import Pool

from lib.env.reward import WeightedUnrealizedProfit

np.warnings.filterwarnings('ignore')


def optimize_code(params):
    from lib.RLTrader import RLTrader

    trader = RLTrader(**params)
    trader.optimize()

    return ""


if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count()
    params = {'n_envs': n_processes, 'reward_strategy': WeightedUnrealizedProfit}

    opt_pool = Pool(processes=n_processes)
    results = opt_pool.imap(optimize_code, [params for _ in range(n_processes)])

    print([result.get() for result in results])

    from lib.RLTrader import RLTrader

    trader = RLTrader(**params)
    trader.train(test_trained_model=True, render_test_env=True, render_report=True, save_report=True)
    # save the trader to disk
    filename = 'Trader_model_model.sav'
    pickle.dump(trader, open(filename, 'wb'))
    print('dumped the model here ')
 
    # some time later...
 
    # load the trader from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    print('complete!')
