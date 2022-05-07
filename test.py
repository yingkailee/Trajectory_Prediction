import os
import sys
import random
import pickle
import copy
import math

from sumolib import checkBinary
import traci

def run(mode, removal_time = 0):
	step = 0
	vehicle_pos = {}
	removed_veh = ''
	timestep_vehs = []

	#while traci.simulation.getMinExpectedNumber() > 0:
	while step <= 1000:
		step += 1
		traci.simulationStep()

		vehicleID = traci.vehicle.getIDList()

		for veh_id in vehicleID:
			if veh_id in vehicle_pos:
				vehicle_pos[veh_id].append([step, traci.vehicle.getPosition(veh_id)])
			else: 
				vehicle_pos[veh_id] = [[step, traci.vehicle.getPosition(veh_id)]]

		if mode == 1:
			if step == removal_time:
				removed_veh = random.choice(vehicleID)
				timestep_vehs = vehicleID
				traci.vehicle.remove(removed_veh)
	traci.close()

	return vehicle_pos, removed_veh, timestep_vehs

# given two trajectories starting at same time, calculate ADE up to 20 timesteps
def ADE(traj1, traj2):
	count = 0
	sum_disp = 0.0
	smaller_len = (len(traj1), len(traj2))[len(traj2) < len(traj1)]

	if(len(traj1) == 0 or len(traj2) == 0):
		return 0.0

	while(traj1[0][0] < traj2[0][0]):
		traj1.pop(0)
		if(len(traj1) == 0 or len(traj2) == 0):
			return 0.0

	while(traj1[0][0] > traj2[0][0]):
		traj2.pop(0)
		if(len(traj1) == 0 or len(traj2) == 0):
			return 0.0

	smaller_len = (len(traj1), len(traj2))[len(traj2) < len(traj1)]

	for i in range(smaller_len):
		count += 1
		xy1 = traj1[i][1]
		xy2 = traj2[i][1]

		sum_disp += math.sqrt((xy1[0] - xy2[0])**2 + (xy1[1] - xy2[1])**2)

		if count == 20:
			break

	if(count == 0):
		return 0.0
	return sum_disp / count

# try using get_data.py

# code ideas - create separate run() for removal and no removal
# change looping logic in main() too

def main(pickle_num = 1):
	sims = 2
	times_removal = 2

	network = ' -n first.net.xml'
	route = ' -r first.rou.xml'
	end = ' -e 1000';

	pickle_name = 'sims' + str(pickle_num) + '.pkl'
	data_file = open(pickle_name,'wb')

	for i in range(sims):
		original_sim = {}
		removed_sim = {}
		removal_veh = 0

		j = 0
		while j <= times_removal:
			if j == 0:
				command = 'randomTrips.py' + network + route + end + ' --random'
				print(command)
				os.system(command)

			traci.start([sumoBinary, "-c", "first.sumocfg", "--tripinfo-output", "tripinfo.xml"])

			if j == 0:
				run_result = run(0)
				original_sim = run_result[0]
					
				pickle.dump(original_sim, data_file)

				timestep_counters = [0] * 1002

				for traj in original_sim.values():
					for time_loc in traj:
						timestep_counters[time_loc[0]] += 1

				max = 0
				max_ind = 0
				ind = 0
				while ind < 1001:
					if timestep_counters[ind] > max:
						max = timestep_counters[ind]
						max_ind = ind
					ind += 1
			else:
				if max_ind < 950:
					removal_time = max_ind
				else:
					removal_time = 950

				run_result = run(1, removal_time)
				removed_sim = run_result[0]
				removed_veh = run_result[1]
				timestep_vehs = run_result[2]

				uncopy_sim1 = copy.deepcopy(original_sim)
				for key in uncopy_sim1:
					i = len(uncopy_sim1[key]) - 1
					while i >= 0:
						if uncopy_sim1[key][i][0] < removal_time:
							del uncopy_sim1[key][i]  
						i = i - 1

				uncopy_sim2 = copy.deepcopy(removed_sim)
				for key in uncopy_sim2:
					i = len(uncopy_sim2[key]) - 1
					while i >= 0:
						if uncopy_sim2[key][i][0] < removal_time:
							del uncopy_sim2[key][i]  
						i = i - 1

				# possibly purge cars in timestep_vehs that don't last for 20 timesteps after removal_time
				timestep_vehs_arr = []
				for veh in timestep_vehs:
					timestep_vehs_arr.append(veh)

				veh_ADE = []
				for veh in timestep_vehs_arr:
					veh_ADE.append(ADE(uncopy_sim1[veh], uncopy_sim2[veh]))

				removals = [removal_time, removed_veh, timestep_vehs_arr, veh_ADE]
				pickle.dump(removals, data_file)

			j += 1

	data_file.close()

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

if __name__ == "__main__":
	sumoBinary = checkBinary('sumo')

	pickle_count = 2

	for pickle_num in range(pickle_count):
		main(pickle_num)