import pickle
from matplotlib import pyplot as plt, colors
import numpy as np
import torch
import matplotlib.cm as cm
from matplotlib import rcParams
import gc
import matplotlib
matplotlib.use('agg')

def scan_file(filename):
    data_file = open(filename, "rb")


    data = pickle.load(data_file)

    start_times, durations, trajectories = [], [], []
    for key, value in data.items():
        start_times.append(value[0][0])
        durations.append(len(value))
        arr = np.zeros((1001, 2))
        for i in range(len(value)):
            #print(value[i][0]-1)
            arr[value[i][0]-1] = value[i][1]

        #print(arr)
        #arr = np.asarray(arr)
        trajectories.append(arr)

    start_times = np.asarray(start_times)
    durations = np.asarray(durations)
    trajectories = np.asarray(trajectories)



    return start_times, durations, trajectories

def vis_scenarios(start_times, durations, trajectories, model, file_id):
    rcParams['savefig.dpi'] = 300
    rcParams.update({'font.size': 12})

    device = torch.device("cpu")
    temporal_resolution = 1
    simulation_duration = 1001
    historical_observations = 20

    min_n_cars = 10

    end_times = start_times+durations

    scenarios = []
    count = 0
    for t in range(1, simulation_duration+1, temporal_resolution):
        t_id = t-1
        cars_exist_long_enough = np.argwhere((t-start_times)>=historical_observations).ravel()
        cars_still_exist = np.argwhere(t<=end_times).ravel()

        possible_candidates = np.intersect1d(cars_exist_long_enough, cars_still_exist)
        if len(possible_candidates)>=min_n_cars:
            # at least two cars together

            current_trajectories = trajectories[possible_candidates, t_id-historical_observations:t_id, :]

            for i in range(len(current_trajectories)):
                main_trajectory = current_trajectories[i]
                other_trajectories = np.delete(current_trajectories, i, 0)

                model_inputs = other_trajectories-np.expand_dims(main_trajectory, 0)

                # get a forward pass
                _input = torch.from_numpy(np.array(model_inputs).astype(np.float32)).to(device)
                model.eval()
                # Encoder
                batch_size = _input.shape[0]
                decoder_outputs = model(_input.view(batch_size, -1))
                decoder_outputs = decoder_outputs.view(-1).cpu().detach().numpy()
                order = np.argsort(decoder_outputs).ravel()

                order = order/len(order)

                cmap = plt.cm.hot
                norm = colors.Normalize(vmin=0, vmax=1.5)

                # Get ranking:

                fig, ax = plt.subplots(figsize=(8,8))
                ax.plot(current_trajectories[i, :, 0], current_trajectories[i, :, 1], c='tab:blue')
                ax.scatter(current_trajectories[i, -1, 0], current_trajectories[i, -1, 1], c='tab:blue')
                for j in range(batch_size):
                    #print(j, np.shape(other_trajectories[j]))
                    ax.plot(other_trajectories[j, :, 0], other_trajectories[j, :, 1], color=cmap(norm(order[j])))
                    ax.scatter(other_trajectories[j, -1, 0], other_trajectories[j, -1, 1], color=cmap(norm(order[j])))
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                cbar = fig.colorbar(sm)
                cbar.ax.set_ylabel("Saliency", labelpad=15, rotation=270)
                plt.axis("off")
                plt.tight_layout()
                plt.gca().set_aspect("equal", adjustable="box")
                plt.savefig("./figures/ranked/"+file_id+"_"+str(count)+".png", format="png")
                plt.close()
                plt.cla()
                plt.clf()

                # Get only most influential:
                fig, ax = plt.subplots()
                ax.plot(current_trajectories[i, :, 0], current_trajectories[i, :, 1], c='tab:blue')
                ax.scatter(current_trajectories[i, -1, 0], current_trajectories[i, -1, 1], c='tab:blue')
                for j in range(batch_size):
                    #print(j, np.shape(other_trajectories[j]))
                    if j != np.argmax(order):
                        c = 'tab:gray'
                    else:
                        c = 'tab:red'
                    ax.plot(other_trajectories[j, :, 0], other_trajectories[j, :, 1], color=c)
                    ax.scatter(other_trajectories[j, -1, 0], other_trajectories[j, -1, 1], color=c)
                plt.axis("off")
                plt.tight_layout()
                plt.gca().set_aspect("equal", adjustable="box")
                plt.savefig("./figures/best/"+file_id+"_"+str(count)+".png", format="png")
                plt.close()
                plt.cla()
                plt.clf()

                count += 1
                gc.collect()

if __name__=="__main__":
    scan_file("./data/sims2.pkl")