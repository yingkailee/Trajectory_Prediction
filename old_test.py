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

def run(mode, removal_time = 300, random_vehicle_ID = 0):
	step = 0
	vehicle_pos = {}
	snapshot = []
	first_vehicle = ''

	#while traci.simulation.getMinExpectedNumber() > 0:
	while step <= 1000:
		step += 1
		traci.simulationStep()

		vehicleID = traci.vehicle.getIDList()
		if len(vehicleID) > 0 and first_vehicle == '':
			first_vehicle = vehicleID[0]
		if len(vehicleID) > 0 and first_vehicle != '' and vehicleID[0] != first_vehicle and len(snapshot) == 0:
			snapshot = vehicleID

		for veh_id in vehicleID:
			if veh_id in vehicle_pos:
				vehicle_pos[veh_id].append([step, traci.vehicle.getPosition(veh_id)])
			else: 
				vehicle_pos[veh_id] = [[step, traci.vehicle.getPosition(veh_id)]]

		if mode == 1:
			if step == removal_time:
				snapshot = vehicleID
				random_vehicle_ID = random.choice(vehicleID)
				traci.vehicle.remove(random_vehicle_ID)

	traci.close()

	return vehicle_pos, random_vehicle_ID, snapshot


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

# initialize array of size 1000, loop through and increment using car timesteps
# get max (prune the too short car frames)
# split into multiple pickle files
# 50 simulations per pickle file
# name them sim1.pkl, sim2.pkl, sim3.pkl, etc
# try using get_data.py

# code ideas - create separate run() for removal and no removal
# change looping logic in main() too

def main():
	sims = 50
	times_removal = 2

	network = ' -n first.net.xml'
	route = ' -r first.rou.xml'
	end = '' #' -e ' + ' ';


	data_file = open('pickle.pkl','wb')

	dict_veh = []

	fails = 0
	for i in range(sims):
		#command = 'randomTrips.py' + network + route + end + ' --random'
		#os.system(command)
		original_sim = {}
		removed_sim = {}
		removal_time = 300
		snapshot = []
		removal_veh = 0
		first_end_step = 0

		j = 0
		while j <= times_removal:
			if j == 0:
				command = 'randomTrips.py' + network + route + end + ' --random'
				os.system(command)
			traci.start([sumoBinary, "-c", "first.sumocfg", "--tripinfo-output", "tripinfo.xml"])
			#print()
			#print(j)
			if j == 0:
				run_result = run(0)
				original_sim = run_result[0]
				snapshot = run_result[2]

				ind = 0
				dict_veh.clear()
				for v in snapshot:
					path = original_sim[snapshot[ind]]
					if(path[0][0] < removal_time - 20 and path[len(path)-1][0] > removal_time + 20):
						dict_veh.append(snapshot[ind])
					ind += 1
				
				try:
					first_vehicle = snapshot[0]
				except:
					#j -= 1
					#print('continue')
					fails += 1
					continue

				first_vehicle_end_step = original_sim[first_vehicle][len(original_sim[first_vehicle]) - 1][0]
				removal_time = first_vehicle_end_step

				if removal_time > 750:
					#j -= 1
					#print('continue')
					fails += 1
					continue
					
				pickle.dump(original_sim, data_file)
			else:
				#print()
				#print(removal_time)
				if first_vehicle_end_step < 750:
					removal_time = random.randint(first_vehicle_end_step, first_vehicle_end_step + 200)

				run_result = run(1, removal_time)
				removed_sim = run_result[0]
				removed_veh = run_result[1]
				snapshot = run_result[2]

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
				for i in snapshot:
					dict_veh.append(i)
				veh_ADE = []
				for veh in dict_veh:
					new_ADE = ADE(uncopy_sim1[veh], uncopy_sim2[veh])
					veh_ADE.append(new_ADE)

				removals = [removal_time, removed_veh, dict_veh, veh_ADE]
				pickle.dump(removals, data_file)

			j += 1

	data_file.close()
	print()
	print(fails)

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