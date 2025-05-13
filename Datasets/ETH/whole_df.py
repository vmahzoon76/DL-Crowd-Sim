#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tug85997

This python script takes the raw .txt file from eth_seq and eth_hotel scenes. Then it preprocesses it and saves as a dataframe
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from datetime import datetime
from matplotlib.pyplot import figure


def raw_to_df(data_folder, save_file_name):
    # Old code for checking if all raw frame number values increase by 6.0
    '''
    frame_number = df['frame_number'].unique()
    frame_number.sort()
    for i in range(frame_number.shape[0]-1):
        a = frame_number[i]
        b = frame_number[i+1]
        if b-a != 6.0:
            print("at idx", i, ':', b, '-', a, '=', b-a)
    '''

    start_time = datetime.now()
    # make a list to save the input/output samples
    sample_list = []

    col_list = ['frame_number', 'pedestrian_ID', 'pos_x', 'pos_z', 'pos_y', 'v_x', 'v_z', 'v_y']

    if not os.path.isfile(data_folder + 'obsmat.txt'):
        print("No obsmat.txt exists in this folder. Exiting program and going back to Main()")
        return

    print("Opening raw data")
    df = pd.read_csv(data_folder + 'obsmat.txt', sep=r"\s+", header=None)
    df.columns = col_list

    # Open the Ped Table
    with open(data_folder + 'Ped_Table.pkl', 'rb') as handle:
        goal_df = pickle.load(handle)

    # Since the README.txt says the z-axis is not used, we can remove it from the dataframes (all of the values are 0.0 for these columns)
    drop_list = ['pos_z', 'v_z']
    df = df.drop(drop_list, axis=1)

    # Make frame_number and pedestrian_ID type int
    df["frame_number"] = df["frame_number"].astype("int")
    df["pedestrian_ID"] = df["pedestrian_ID"].astype("int")

    # Get unique list of the frame numbers. Map them integers starting at 0 for easier
    frames_unique = list(df.frame_number.unique())
    new_frames_unique = list(range(0, len(frames_unique)))
    frame_to_num = {}
    count = 0
    for number in frames_unique:
        frame_to_num[number] = new_frames_unique[count]
        count = count + 1
    df["frame_number"] = df["frame_number"].map(frame_to_num)

    # Merge the goal_df to the raw df by pedestrian_ID. So we can have the goals of the agents
    df = pd.merge(df, goal_df, on='pedestrian_ID', how='left')

    # Make a list of the pedestrian IDs so we can loop through them
    pedestrian = list(df["pedestrian_ID"].unique())
    pedestrian.sort()
    for ped in pedestrian:

        # get the data, goals, and first/last frames for a pedestrian
        this_ped = df[df["pedestrian_ID"] == ped]
        first_frame = this_ped['First_Frame'].iloc[0]
        last_frame = this_ped['Last_Frame'].iloc[0]

        print(" Total samples:", len(sample_list), ", Loop for ped:", ped, "is in frames", first_frame, "to",
              last_frame, "adding", (last_frame - first_frame), "samples")

        # loop from first to last frame - 1 (The -1 is to  account that we need the the next consecutive frame for the outputted velocity)
        for idx in range((last_frame - first_frame)):
            # initialize 2 dataframes to save the input and output for the data sample
            '''
            The sample_list should be a list where len(sample_list) = number of generated simulations.
                Each element of sample_list should be another list of 2 elements:
                    0. input dataframe, the columns are the (x, y) values for
                        - current locations, 
                        - current velocities, and 
                        - goal locations 
                        of all the agents in a scene.
                    1. output dataframe, the columns are the (x,y) values for the 
                        next-timestep velocity of the first agent listed in input 
                        dataframe.
                        - (Since it is only 2 values, we could make this into
                        an array or list. But I kept it as a dataframe to keep the 
                        consistency between the generated and real life data)
            '''
            input_col_list = ['pedestrian_ID', 'curr_loc_x', 'curr_loc_y', 'curr_vel_x', 'curr_vel_y', 'goal_loc_x',
                              'goal_loc_y']
            output_col_list = ['pedestrian_ID', 'pred_vel_x', 'pred_vel_y']
            input_df = pd.DataFrame(columns=input_col_list)
            output_df = pd.DataFrame(columns=output_col_list)

            x_frame = first_frame + idx
            y_frame = x_frame + 1

            # get the agents' info from the x_frame
            x_peds_col_list = list(df.columns)[1:6] + list(df.columns)[8:]
            x_peds = df[df["frame_number"] == x_frame][x_peds_col_list]

            # move the ped to the first row
            x_peds_top = x_peds[x_peds["pedestrian_ID"] == ped]
            x_peds_bottom = x_peds[x_peds['pedestrian_ID'] != ped]
            x_peds = pd.concat([x_peds_top, x_peds_bottom], ignore_index=True)

            # rename the columns to match the columns in input_df
            x_peds.columns = input_col_list

            # add all the agents' info to the input dataframe
            input_df = pd.concat([input_df, x_peds], ignore_index=True)

            # get the ped's next-timestep velocity in y_frame
            y_peds_col_list = list(df.columns)[1:2] + list(df.columns)[4:6]
            y_peds = df[(df["frame_number"] == y_frame) & (df["pedestrian_ID"] == ped)][y_peds_col_list]

            # rename the columns to match the columns in output_df
            y_peds.columns = output_col_list

            # add the agent of interest (variable in this For-loop is ped)'s velocity of the y_frame to the output dataframe
            output_df = pd.concat([output_df, y_peds], ignore_index=True)

            # add the input/output dataframe to the sample_list
            sample_list.append([x_frame,input_df, output_df])
    sample_list = sorted(sample_list, key=lambda x: x[0])
    for inner in sample_list:
        del inner[0]
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(sample_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    print("done! Checking file integrity...")
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)

    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    '''
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", len(sample_list), "samples")
    return sample_list


def make_video():
    return


if __name__ == '__main__':
    data_folder = '../ETH/'
    save_file_name = 'whole_eth'
    df = raw_to_df(data_folder, data_folder + save_file_name)

    # old code for finding the max number of agents in a single frame
    '''
    print("loaded dataframe")

    # We want to see the max number of agents that are in a single timeframe
    most_crowded_frame = -999
    max_agents = -999
    unique_timeframes = list(df['frame_number'].unique())

    for i in range(0, len(unique_timeframes)):
        this_frame = df[df['frame_number']== i]
        unique_agents = list(this_frame['pedestrian_ID'].unique())
        agent_count = len(unique_agents)

        if agent_count > max_agents:
            print('updating max from', max_agents, 'to', agent_count)
            most_crowded_frame = i
            max_agents = agent_count
    print("max_agents: ", max_agents, "in frame_number:", most_crowded_frame)
    '''

    # checking the generated pkl file
    with open(data_folder + save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    print("hi hello")