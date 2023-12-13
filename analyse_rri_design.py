from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

exp_mapping = {
    '17753415': 'No Info',
    '17753420': 'No Info_random',  # all_nucs_and_structure',
    '17753425': 'Full Structure',
    '17753429': 'Full Structure Random',
    '17753431': 'Structure and Sequence',
    '17753432': 'Structure and Sequence Random',  #'all_nucs_and_structure_random',
}

def process_file(file_path, chunk_size):
    df = pd.read_csv(file_path, sep='\s+', names=['ID', 'Time', 'Reward', 'Sequence', 'Structure'])
    if df.shape[0] < 100000:
        return None
    print(df)

    df.loc[:, 'N'] = df['Sequence'].apply(lambda x: any([i == 'N' for i in x]))
    print(file_path.stem, df['N'].sum() / df.shape[0])

    # Mapping experiment ID and seed
    experiment_id = exp_mapping[str(file_path.stem).split('[')[0]]
    seed = int(str(file_path.stem).split('[')[1].split(']')[0])
    df = df[:100000]

    # Calculating energy by 3rd root of reward since reward was computed as energy³ in the code
    df.loc[:, 'Energy'] = df['Reward'].apply(lambda x: -1 * (x**(1/3)))
    # df['Reward'] = df['Reward'].astype('float').replace(-200, 0.0)

    # Chunking data every N steps and calculating mean and std dev
    df['chunk'] = df.index // chunk_size
    chunked = df.groupby('chunk')['Energy'].agg(['mean', 'std']).reset_index()
    chunked['experiment_id'] = experiment_id
    chunked['seed'] = seed
    return chunked


paths = list(Path('rri_results').glob('177534*.o'))

all_results = []
chunk_size = 1000  # Change this to your desired chunk size
for p in paths:
    print('### Processing file {} ###'.format(p))
    chunked_data = process_file(p, chunk_size)
    if chunked_data is not None:
        all_results.append(chunked_data)

# Combine all results into a single DataFrame
combined_df = pd.concat(all_results)

# Aggregate across seeds for each experiment ID
final_agg = combined_df.groupby(['experiment_id', 'chunk']).agg({'mean': ['mean', 'std'], 'std': ['mean', 'std']}).reset_index()

for exp_id in final_agg['experiment_id'].unique():

    exp_data = final_agg[final_agg['experiment_id'] == exp_id]
    
    # Extracting mean and std deviation
    mean_reward = exp_data['mean', 'mean']
    std_dev = exp_data['mean', 'std']
    chunks = exp_data['chunk']

    # Plotting the mean reward
    plt.plot(chunks, mean_reward, label=f'Experiment {exp_id}')

    # Adding the std deviation as a shaded area
    plt.fill_between(chunks, mean_reward - std_dev, mean_reward + std_dev, alpha=0.3)

# plt.xscale('log')  # For logarithmic x-axis
# plt.yscale('log')  # For logarithmic y-axis (uncomment this line if needed)

plt.xlabel('Step (in chunks of {})'.format(chunk_size))
plt.ylabel('Average Energy')
plt.title('Mean Energy with Standard Deviation as Confidence Bounds')
plt.legend()
plt.show()