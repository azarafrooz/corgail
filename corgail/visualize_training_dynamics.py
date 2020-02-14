from baselines.common import plot_util as pu

LOG_DIRS = 'logs/coinrun_500_level/'

results = pu.load_results(LOG_DIRS)

smooth_step = 50.0

fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False, smooth_step=smooth_step)
pu.plt.savefig('coinrun_500_level')




