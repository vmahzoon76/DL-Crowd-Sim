import numpy as np
import pandas as pd
from lidar_obs import lidar
import os
import pickle

def get_data(data_folder):
    file_list = os.listdir(data_folder)
    data=[[],[]]
    x = []
    y = []
    count=0

    for file in file_list:

        # check if file is .pkl
        if file.endswith("whole_hotel.pkl"):
            print(file)

            # open .pkl file as sample_list
            '''
            The sample_list should be a list where len(sample_list) = number of generated simulations.
            Each element of sample_list should be another list of 4 elements:
                - current location dataframe
                - current velocity dataframe
                - preferred velocity dataframe
                - preferred velocity dataframe (normalized)
            '''
            load_file = data_folder + "/" + file
            with open(load_file, 'rb') as handle:
                sample_list = pickle.load(handle)

            # for each sample in sample_list, take what you need for x(input) and y (output)
            '''
            For this model, the inputs for each agent are:
                - current location
                - current velocity
                - preferred velocity

            (Please note that we do not use the normalized preferred velocity dataframe)

            and the output is:
                - current velocity
            '''
            for sample_idx in range(len(sample_list)):
                count+=1
                if count%500==0:
                    print(count)
                this_sample = sample_list[sample_idx]

                curr_loc_df = this_sample[0][["curr_loc_x", "curr_loc_y"]]
                curr_vel_df = this_sample[0][["curr_vel_x", "curr_vel_y"]]
                pref_vel_df = this_sample[0][["goal_loc_x", "goal_loc_y"]]
                pos=np.array(curr_loc_df.iloc[0])
                pref_vel_df = pref_vel_df - np.array(curr_loc_df.iloc[0])
                curr_loc_df = curr_loc_df - curr_loc_df.iloc[0]

                curr=np.array(curr_loc_df).tolist()
                curr_lidar=curr[1:]
                prev=np.array(curr_loc_df) - 0.4 * np.array(curr_vel_df) + 0.4 * np.array(curr_vel_df)[0]
                prev_lidar=prev.tolist()[1:]

                goal=np.array(pref_vel_df)[0].tolist()
                curr_vel=np.array(curr_vel_df)[0].tolist()
                bench_corrds = [[-0.517, -10.065],
                                [-0.613, -7.755],
                                [-1.199, -7.737],
                                [-1.200, -10.015]]
                hex1 = [[-0.827, -5.126],
                        [-0.89199, -5.0134],
                        [-1.022, -5.01343],
                        [-1.087, -5.126],
                        [-1.022, -5.23858],
                        [-0.89199, -5.23859]]
                hex2 = [[-0.689, -1.76],
                        [-0.754, -1.6474],
                        [-0.88399, -1.6475],
                        [-0.949, -1.76],
                        [-0.88399, -1.87258],
                        [-0.754, -1.872583]]
                hex3 = [[-0.727, 1.917],
                        [-0.792, 2.02958],
                        [-0.92199, 2.02957],
                        [-0.987, 1.917],
                        [-0.92199, 1.8044],
                        [-0.792, 1.80441]]
                obs_orig = []
                obs_orig.append(hex1)
                obs_orig.append(hex2)
                obs_orig.append(hex3)
                obs_orig.append(bench_corrds)
                obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()] + [(np.array(obs_orig[2]) - pos).tolist()] + [(np.array(obs_orig[3]) - pos).tolist()]
                c=lidar(10,0.5,obs ,curr_lidar)
                c.sense_obstacles()
                #c.plot()
                curr_lidar=c.scan_array
                pos=pos-0.4*np.array(curr_vel_df.iloc[0])
                obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()] + [(np.array(obs_orig[2]) - pos).tolist()] + [(np.array(obs_orig[3]) - pos).tolist()]
                c = lidar(10, 0.5, obs, prev_lidar)
                c.sense_obstacles()
                prev_lidar = c.scan_array
                l=prev_lidar + curr_lidar + curr_vel + goal

                # reshape the arrays into (#sample, #agents, #components)


                y_curr_vel = np.array(this_sample[1][["pred_vel_x", "pred_vel_y"]]).reshape(2,)
                y=y_curr_vel.tolist()
                # append the sample to the x and y lists
                data[0].append(l)
                data[1].append(y)

    return data


data= get_data('transformer/')
save_file_name="whole_hotel"
with open(save_file_name + '.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
