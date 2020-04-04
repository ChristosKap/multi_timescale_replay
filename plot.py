import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dirs = ['./mtr_test/']
labels = ['MTR']


dfs = []
eval_gravities = [-7, -9.5, -12, -14.5, -17] 
eval_interp_interval = 1000
train_interp_interval = 1000
num_evals = 5
num_timesteps = 5000000
eval_ma = 100
train_ma = 100
axes_font_size = 16

for i in range(len(dirs)):
    dfs.append(pd.read_csv(dirs[i]+"progress.csv"))
    
# Training reward
plt.figure()
for i in range(len(dirs)):
    df = dfs[i]
    interp_rews = np.array(np.interp(range(0, num_timesteps, train_interp_interval),
                             df['total timesteps'],
                             df['mean 100 episode reward'].rolling(1).mean()))
    print(len(interp_rews))
    plt.plot(range(0, num_timesteps, train_interp_interval), pd.Series(interp_rews).rolling(train_ma).mean(), label=labels[i])
plt.xlabel('Timesteps', size=axes_font_size)
plt.ylabel('Training Reward', size=axes_font_size)
plt.legend()

# Individual evaluation rewards
for idx, df in enumerate(dfs):
    if 'eval_0 ep_rewmean' in df:
        plt.figure()
        interp_rews = np.array([np.interp(range(0, num_timesteps, eval_interp_interval),
                             df['total timesteps'],
                             df['eval_'+str(j)+' ep_rewmean'].rolling(1).mean()) for j in range(num_evals)])
        for j in range(num_evals):
            plt.plot(range(0, num_timesteps, eval_interp_interval), pd.Series(interp_rews[j,:]).rolling(eval_ma).mean(), label=eval_gravities[j])
        plt.xlabel('Timesteps', size=axes_font_size)
        plt.ylabel('Evaluation Reward', size=axes_font_size)
        plt.legend(title='Gravity')

# Mean evaluation reward
plt.figure()
for idx, df in enumerate(dfs):
    if 'eval_0 ep_rewmean' in df:
        interp_rews = np.array([np.interp(range(0, num_timesteps, eval_interp_interval),
                             df['total timesteps'],
                             df['eval_'+str(j)+' ep_rewmean'].rolling(1).mean()) for j in range(num_evals)])
        plt.plot(range(0, num_timesteps, eval_interp_interval), pd.Series(np.mean(interp_rews, axis=0)).rolling(eval_ma).mean(), label=labels[idx])
plt.xlabel('Timesteps', size=axes_font_size)
plt.ylabel('Mean Evaluation Reward', size=axes_font_size)
plt.legend()

plt.show()
