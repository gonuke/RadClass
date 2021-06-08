import numpy as np
import pandas as pd
import math


class BackgroundEstimator:
    background = pd.DataFrame(columns=['timestamp', 'count-rate'])
    data = np.empty((0, 2))
    # sample_size = None

    def __init__(self, confidence=0.95, sample_size=None):
        self.confidence = confidence
        self.sample_size = sample_size

    def save_all(self, observations):
        data = pd.DataFrame()
        data['timestamps'] = observations[:, 0]
        for i in range(1000):
            data[str(i+1)] = observations[:, i+1]
        data.to_csv('results/oct2019_channels.csv', mode='a')

    def add_obs(self, timestamp, obs):
        count_rate = np.sum(obs)
        dfrow = {'timestamp': timestamp, 'count-rate': count_rate}
        self.background.loc[len(self.background.index)] = dfrow

        # using numpy for building instead of pandas
        # self.data = np.vstack([self.data, [timestamp, count_rate]])

        if self.sample_size is not None:
            self.estimate()

    def sort(self):
        return self.background.sort_values(by='count-rate')

    def estimate(self):
        # building pandas DataFrame once from numpy array
        # self.background['timestamp'] = self.data[:,0]
        # self.background['count-rate'] = self.data[:,1]
        # find number of background samples
        if self.sample_size is None:
            print('Not working :(')
            total = len(self.background.index)
        elif self.sample_size is not None:
            total = self.sample_size
        num_samples = math.floor(total * (1-self.confidence))

        self.background = self.sort()

        return self.background.iloc[:num_samples]
