import pickle
from matplotlib import pyplot as plt
import numpy as np

def read_file(filename, show=False):
    data_file = open(filename, "rb")

    #return cars, removed_ades
    n_removed = 2
    count = 0
    data = []
    labels = []
    trajectories = None
    removed_info = []
    hist_len = 20
    while 1:
        try:
            x = pickle.load(data_file)
            if count == 0:
                trajectories = x
            else:
                removed_info.append(x)
            if count >= n_removed:
                # process data here:

                for i in range(n_removed):
                    # go through info about each removed car:
                    removed_time = removed_info[i][0]  # timestep at which a car was removed
                    removed_id = removed_info[i][1] # id of removed car
                    cars_ids = removed_info[i][2]   # ids of cars to check
                    ades = removed_info[i][3]       # ADE information
                    for j in range(len(cars_ids)):
                        if cars_ids[j] != removed_id:
                            #print(removed_id, cars_ids[j])

                            idx1 = np.argwhere(np.array(np.array(trajectories[removed_id])[:, 0]) == removed_time)[0][0]
                            idx2 = np.argwhere(np.array(np.array(trajectories[cars_ids[j]])[:, 0]) == removed_time)[0][0]

                            if idx1 < hist_len or idx2 < hist_len:
                                # make sure that only relevant cars are passed through
                                continue
                            n = np.array(np.array(trajectories[removed_id])[(idx1-hist_len):idx1, 1])
                            m = np.array(np.array(trajectories[cars_ids[j]])[(idx2-hist_len):idx2, 1])

                            x = np.asarray([n[i][0] for i in range(len(n))])
                            y = np.asarray([n[i][1] for i in range(len(n))])
                            carA = np.transpose(np.vstack((x, y)))
                            if show:
                                plt.scatter(x, y, c='b')
                            x = np.asarray([m[i][0] for i in range(len(m))])
                            y = np.asarray([m[i][1] for i in range(len(m))])
                            carB = np.transpose(np.vstack((x, y)))
                            data.append(carA-carB)
                            labels.append(ades[j])
                            if show:
                                plt.scatter(x, y, c='r')
                                plt.show()



                # re-initialize values
                count = 0
                trajectories = None
                removed_info = []
            else:
                count+=1
        except EOFError:
            break
    return np.asarray(data), np.asarray(labels)

if __name__=="__main__":
    data = read_file("./data/data1.pkl", show=True)
