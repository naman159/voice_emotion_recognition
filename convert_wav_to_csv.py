"""
Instructions:
* Use the following command to convert a file.wav to file.csv

Command:
* python convert_wav_to_csv.py --input file.wav

Extra considerations:
* You need to replace file to something else (ex. speaker-data-Tom-1.wav)
"""

import sys, os, os.path
from scipy.io import wavfile
import pandas as pd
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input .wav file name')
args = parser.parse_args()

def reformat_csv(filename, outfile):
    label = int(filename[-10])
    index = []
    data = []
    with open(filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            try:
                index.append(int(row[0]))
                data.append(int(float(row[1])))
            except:
                print("Avoiding the header")
    # Convert data into slices of 8000 slices
    nr_slice = len(data)//8000
    sliced_data = []
    idx = 0
    for _ in range(nr_slice):
        temp = []
        for _ in range(8000):
            temp.append(data[idx])
            idx += 1
        temp.append(label)
        sliced_data.append(list(temp))
    sliced_data_np = np.array(sliced_data)
    #print(sliced_data_np.shape)
    if not os.path.exists('data'):
        os.makedirs('data')
    pd.DataFrame(sliced_data_np).to_csv("data/" + outfile, header=False)

def main():
    input_filename = args.input
    if input_filename[-3:] != 'wav':
        print('WARNING!! Input File format should be *.wav')
        sys.exit()
    samrate, data = wavfile.read(str(input_filename))
    print('Load is Done! \n')
    wavData = pd.DataFrame(data)
    if len(wavData.columns) == 2:
        print('Stereo .wav file found\n')
        wavData.columns = ['R', 'L']
        #stereo_R = pd.DataFrame(wavData['R'])
        stereo_L = pd.DataFrame(wavData['L'])
        print('Saving the left stereo...\n')
        #stereo_R.to_csv(str(input_filename[:-4] + "_Output_stereo_R.csv"), mode='w')
        stereo_L.to_csv(str(input_filename[:-4] + "_temp.csv"), mode='w')
        # wavData.to_csv("Output_stereo_RL.csv", mode='w')
    elif len(wavData.columns) == 1:
        print('Mono .wav file found\n')
        wavData.columns = ['M']
        wavData.to_csv(str(input_filename[:-4] + "_temp.csv"), mode='w')
    else:
        print('Multi channel .wav file found\n')
        print('number of channel : ' + len(wavData.columns) + '\n')
        wavData.to_csv(str(input_filename[:-4] + "_temp.csv"), mode='w')
    reformat_csv(str(input_filename[:-4] + "_temp.csv"), str(input_filename[:-4] + ".csv"))
    print("Saving done")


if __name__ == "__main__":
    main()