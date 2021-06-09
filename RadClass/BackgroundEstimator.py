import numpy as np
import pandas as pd
import math


class BackgroundEstimator:
    '''
    Integrates a gross count rate from a spectrum and orders samples in
        ascending order of magnitude. Saves a certain percentage samples
        to file with corresponding gross count rate.

    Attributes:
    background: Pandas DataFrame used to store timestamps and count rates.
    data: Numpy array used to incrementally store data throughout analysis
        before sending/saving in background.
    confidence: Percentage of samples to disregard. 1-confidence number of
        samples are saved. For example, if confidence=0.95, the lowest 5% of
        count rates/timestamps are saved.
    ofilename: Output filename excluding extension. File is stored as a csv.
        Default is bckg_results.
    '''

    background = pd.DataFrame(columns=['timestamp', 'count_rate'])
    data = np.empty((0, 2))

    def __init__(self, confidence=0.95, ofilename='bckg_results'):
        self.confidence = confidence
        self.ofilename = ofilename

    def __del__(self):
        '''
        Saves results upon completion of script, even if the script crashes.
        '''

        self.estimate()
        # this method could be compressed to save memory
        self.background.to_csv(self.ofilename+'.csv', index=False)

    def save_all(self, observations):
        data = pd.DataFrame()
        data['timestamps'] = observations[:, 0]
        for i in range(1000):
            data[str(i+1)] = observations[:, i+1]
        data.to_csv('results/oct2019_channels.csv', mode='a')

    def sort(self):
        '''
        Wrapper function for Pandas DataFrame sort_values method. Organizes
            samples in ascending order of gross count rate.
        '''

        return self.background.sort_values(by='count_rate', ascending=True)

    def estimate(self):
        '''
        Saves data to a Pandas DataFrame and selects a subset with the smallest
            gross count rates.

        Attributes:
        samples_size: The total number of samples analyzed.
        num_samples: The number of samples to select and save.
        '''
        # building pandas DataFrame once from numpy array
        self.background['timestamp'] = self.data[:, 0]
        self.background['count_rate'] = self.data[:, 1]

        # find number of background samples
        sample_size = len(self.background.index)
        # +1 to account for indexing
        num_samples = math.floor(sample_size * (1-self.confidence)) + 1

        # sort and separate the smallest count-rates
        self.background = self.sort()
        self.background = self.background.iloc[:num_samples]

    def run(self, data, timestamp):
        '''
        Method called by parent class during analysis. Integrates/collects a
            gross count rate and stores it in an iterim NumPy array.

        Attributes:
        count_rate: gross count rate from individual observation.
        '''

        count_rate = np.sum(data)

        # alternative: use pandas and "append"
        # slower (by 3 and 20 minutes respectively) than storing in array
        # dfrow = {'timestamp': timestamp, 'count-rate': count_rate}
        # self.background.loc[len(self.background.index)] = dfrow
        # self.background = pd.concat([self.background, pd.DataFrame([data], index=[timestamp])])

        # using numpy to build lightweight matrix instead of pandas
        self.data = np.vstack([self.data, [timestamp, count_rate]])

        # instead of estimating at the end, checkpointing could occur
        # if len(self.data) % 10000:
        #    self.estimate()

    def write(self):
        '''
        Identical to destructor but can be manually called.
        NOTE: Only write or __del__ should be kept, but there is a design
            question of which is the appropriate method to write to file.
        '''

        self.estimate()
        # this method could be compressed to save memory
        self.background.to_csv(self.ofilename+'.csv', index=False)
