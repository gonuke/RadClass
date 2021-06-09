import numpy as np
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
from RadClass.BackgroundEstimator import BackgroundEstimator
from tests.create_file import create_file


def test_estimation():
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)

    energy_bins = 1000
    timesteps = 100

    timestamps = np.arange(start_date,
                           start_date + (timesteps * delta),
                           delta).astype('datetime64[s]').astype('float64')

    livetime = 0.9
    live = np.full((len(timestamps),), livetime)

    # randomly order incremental "spectra"
    values = np.arange(timesteps)
    np.random.shuffle(values)
    spectra = np.empty((timesteps, energy_bins))
    for i in range(timesteps):
        spectra[i].fill(values[i])
    print(spectra)
    # create sample test file with above simulated data
    create_file(filename, datapath, labels, live, timestamps, spectra,
                timesteps, energy_bins)

    stride = 1
    integration = 1

    confidence = 0.95
    ofilename = 'bckg_results'
    bckg = BackgroundEstimator(confidence=confidence, ofilename=ofilename)
    # run handler script
    classifier = RadClass(stride, integration, datapath,
                          filename, store_data=True, analysis=bckg)
    classifier.run_all()

    bckg.write()

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = 0.0
    results = np.genfromtxt('bckg_results.csv', delimiter=',', skip_header=1)
    print(results)
    np.testing.assert_equal(results[0][1], expected)
    time_idx = np.where(values == 0)[0][0]
    np.testing.assert_equal(results[0][0], timestamps[time_idx])

    expected_num = int((timesteps / integration) * (1 - confidence))
    np.testing.assert_equal(len(results), expected_num)

    os.remove(filename)
    os.remove('bckg_results.csv')

def test_spectral_storage():
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)

    energy_bins = 1000
    timesteps = 100

    timestamps = np.arange(start_date,
                           start_date + (timesteps * delta),
                           delta).astype('datetime64[s]').astype('float64')

    livetime = 0.9
    live = np.full((len(timestamps),), livetime)

    # randomly order incremental "spectra"
    values = np.arange(timesteps)
    np.random.shuffle(values)
    spectra = np.empty((timesteps, energy_bins))
    for i in range(timesteps):
        spectra[i].fill(values[i])
    print(spectra)
    # create sample test file with above simulated data
    create_file(filename, datapath, labels, live, timestamps, spectra,
                timesteps, energy_bins)

    stride = 1
    integration = 1

    confidence = 0.95
    ofilename = 'bckg_results'
    bckg = BackgroundEstimator(confidence=confidence, ofilename=ofilename,
                               store_all=True, energy_bins=energy_bins)
    # run handler script
    classifier = RadClass(stride, integration, datapath,
                          filename, store_data=True, analysis=bckg)
    classifier.run_all()

    bckg.write()

    expected = np.zeros((energy_bins,))
    results = np.genfromtxt('bckg_results.csv', delimiter=',', skip_header=1)
    np.testing.assert_equal(results[0][2:], expected)

    os.remove(filename)
    os.remove('bckg_results.csv')
