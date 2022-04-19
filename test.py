import os
import sys
import random
import optparse
import pickle
import copy
import math

from sumolib import checkBinary
import traci

def remove_dict_val(d):
	for key in d:
		i = len(d[key]) - 1
		while i >= 0:
			if d[key][i][0] < 70:
				del d[key][i]    
			i = i - 1

	keys = list(d)
	for i in range(len(d)):
		if len(d[keys[i]]) == 0:
			del d[keys[i]]

def append_files(data_file, file1):
	file = open(file1, 'rb')
	pickle.dump(pickle.load(file), data_file)

def run(mode, removal_time = 70, random_vehicle_ID = 0):
	step = 0
	vehicle_pos = {}
	snapshot = []

	#while traci.simulation.getMinExpectedNumber() > 0:
	while step <= 1000:
		step += 1
		traci.simulationStep()

		if (step % 1) == 0:
			vehicleID = traci.vehicle.getIDList()

			for veh_id in vehicleID:
				if veh_id in vehicle_pos:
					vehicle_pos[veh_id].append([step, traci.vehicle.getPosition(veh_id)])
				else: 
					vehicle_pos[veh_id] = [[step, traci.vehicle.getPosition(veh_id)]]

			if step == 1000:
				snapshot = vehicleID

		if mode == 1:
			if step == removal_time:
				if random_vehicle_ID == 0:
					random_vehicle_ID = random.choice(vehicleID)
					#print("VEHICLE REMOVED")
					print(random_vehicle_ID)
					traci.vehicle.remove(random_vehicle_ID)
					sys.stdout.flush()


			

	#if mode == 1:
		#remove_dict_val(vehicle_pos)
	print(step)
	traci.close()
	sys.stdout.flush()

	return vehicle_pos, random_vehicle_ID, snapshot

"""
simulation of 5 cars {1, 2, 3, 4, 5}
    [pickle] dict of locations, remove locations after rm_time (not with original dict)
    random 3 cars to remove {2,3,4}
        car removed: 2
        every other car being examined
        {1,3,4,5}

        remove car if doesn't exist from [rm_time - timesteps, rm_time + timesteps]

        new vector of cars being examined
        {1,3}
        corresponding to each is a vector of ADE of old(a) vs new(b) trajectory}
        {1a ADE 1b, 3a ADE 3b}
        
        [pickle] 2, {1,3}, {1a ADE 1b, 3a ADE 3b}
"""

"""
Pickle file
dict of locations before time step
    // [removed car, cars to check, ades of cars being checked]
    [2, {1,3}, {1a ADE 1b, 3a ADE 3b}]
    [3, {1,2}, {1a ADE 1b, 2a ADE 2b}]
    [4, {1,2,3}, {1a ADE 1b, 2a ADE 2b, 3a ADE 3b}]
repeat?
"""

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

# just make sure there's cars with full history of time steps

# 1 pickle file
# 2 diff random simulations, 2 diff cars removed each

# share code on how to extract pickle file again

def main():
	sims = 1
	times_removal = 1

	network = ' -n first.net.xml'
	route = ' -r first.rou.xml'
	end = '' #' -e ' + ' ';

	data_file = open('pickle.pkl','wb')

	for i in range(sims):
		commandy = 'randomTrips.py' + network + route + end + ' --random'
		os.system(commandy)
		original_sim = {}
		removed_sim = {}
		removal_time = 1000

		for j in range(times_removal + 1):
			traci.start([sumoBinary, "-c", "first.sumocfg", "--tripinfo-output", "tripinfo.xml"])

			if j == 0:
				run_result = run(0)
				original_sim = run_result[0]
				snapshot = run_result[2]
				"""print(snapshot)
				for car in snapshot:
					print(original_sim[car])"""

				for car in original_sim:
					print(len(original_sim[car]))

				#print(original_sim)
			else:
				run_result = run(1, removal_time)
				removed_sim = run_result[0]
				removed_veh = run_result[1]

		copy_sim = copy.deepcopy(original_sim)
		for key in copy_sim:
			i = len(copy_sim[key]) - 1
			while i >= 0:
				if copy_sim[key][i][0] >= removal_time:
					del copy_sim[key][i]  
				i = i - 1

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

		dict_veh = []
		"""
		for i in removed_sim.keys():
			if i != removed_veh and len(removed_sim[i]) != 0:
				if removed_sim[i][0][0] < removal_time - 20 and removed_sim[i][len(removed_sim[i]) - 1][0] > removal_time + 20:
						dict_veh.append(i)

						if len(dict_veh) == 2:
							break"""

		#print(original_sim)
	#	print(original_sim[removed_veh])
		#print(dict_veh)
		#print(removed_sim)
	#	print(removed_sim[removed_veh])
		#print(removed_veh)

		veh_ADE = []
		for veh in dict_veh:
			new_ADE = ADE(uncopy_sim1[veh], uncopy_sim2[veh])
			veh_ADE.append(new_ADE)

		data_file = open('pick.pkl', 'wb')
		pickle.dump(copy_sim, data_file)
		removals = [removed_veh, dict_veh, veh_ADE]
		pickle.dump(removals, data_file)
		data_file.close()

		data_file = open('pick.pkl', 'rb')
		orig = pickle.load(data_file)
		remov = pickle.load(data_file)
		data_file.close()

		#print(orig)
		#print(remov)

def get_options():
	opt_parser = optparse.OptionParser()
	opt_parser.add_option("--nogui", action="store_true",
						 default=True, help="run the commandline version of sumo")
	options, args = opt_parser.parse_args()
	return options

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

if __name__ == "__main__":
	options = get_options()
	if options.nogui:
		sumoBinary = checkBinary('sumo')
	else:
		sumoBinary = checkBinary('sumo-gui')

	main()