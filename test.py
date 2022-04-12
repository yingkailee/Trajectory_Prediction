import os
import sys
import random
import optparse
import pickle
import copy
import math

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

def get_options():
	opt_parser = optparse.OptionParser()
	opt_parser.add_option("--nogui", action="store_true",
						 default=True, help="run the commandline version of sumo")
	options, args = opt_parser.parse_args()
	return options

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

def run(mode):
	step = 0
	vehicle_pos = {} # Key: Vehicle ID, Value: 2D list of time and position pairs

	while traci.simulation.getMinExpectedNumber() > 0:
		step += 1
		traci.simulationStep()

		if (step % 50) == 0:
			vehicleID = traci.vehicle.getIDList()

			for veh_id in vehicleID:
				if veh_id in vehicle_pos:
					vehicle_pos[veh_id].append([step//10, traci.vehicle.getPosition(veh_id)])
				else: 
					vehicle_pos[veh_id] = [[step//10, traci.vehicle.getPosition(veh_id)]]

		#remove a vehicle at timestep
		if mode == 1:
			if step == 700:
				random_vehicle_ID = random.choice(vehicleID)
				traci.vehicle.remove(random_vehicle_ID)
				sys.stdout.flush()
			

	#if mode == 1:
		#remove_dict_val(vehicle_pos)

	traci.close()
	sys.stdout.flush()




	return vehicle_pos

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




"""
simulation of X cars {1, 2, 3, X}
[pickle] dict of locations, remove locations after rm_time (not with original dict)

remove 2
create ADE of 1's original trajectory, trajectory after removal
[pickle] 2, {1}, 1 ADE
"""

"""
PICKLE FILE
dict of locations before time step
[2, {1}, 1's ADE]
"""

def ADE(traj1, traj2):
	count = 0
	sum_disp = 0.0
	for i in range(len(traj1)):
		count += 1
		xy1 = traj1[i][1]
		xy2 = traj2[i][1]

		sum_disp += math.sqrt((xy1[0] - xy2[0])**2 + (xy1[1] - xy2[1])**2)

	return sum_disp / count


def main():
	lowerBound = 80
	upperBound = 100
	n = random.randint(lowerBound, upperBound)

	sims = 1
	times_removal = 1

	network = ' -n first.net.xml'
	route = ' -r first.rou.xml'
	end = ' -e ' + str(n);

	data_file = open('pickle.pkl','wb')

	for i in range(sims):
		commandy = 'randomTrips.py' + network + route + end + ' --random'
		os.system(commandy)
		original_sim = {}
		removed_sim = {}
		for j in range(times_removal + 1):
			traci.start([sumoBinary, "-c", "first.sumocfg",
							 "--tripinfo-output", "tripinfo.xml"])
			if j == 0:
				original_sim = run(0)
			else:
				removed_sim = run(1)

			#append_files(data_file, 'picklesub.pkl')

		copy_sim = copy.deepcopy(original_sim)
		for key in copy_sim:
			i = len(copy_sim[key]) - 1
			while i >= 0:
				if copy_sim[key][i][0] >= 70:
					del copy_sim[key][i]  
				i = i - 1


		data_file = open('pick.pkl', 'wb')
		pickle.dump(copy_sim, data_file)
		data_file.close()

		data_file = open('pick.pkl', 'rb')
		original1 = pickle.load(data_file)
		data_file.close()

		traj1 = [[180, (10.0, 15.0)], [185, (20.0, 25.0)]]
		traj2 = [[180, (15.0, 20.0)], [185, (25.0, 30.0)]]

		print(ADE(traj1, traj2))


if __name__ == "__main__":
	options = get_options()
	if options.nogui:
		sumoBinary = checkBinary('sumo')
	else:
		sumoBinary = checkBinary('sumo-gui')

	main()