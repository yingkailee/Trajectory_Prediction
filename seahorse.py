import os
import sys
import optparse
import random
import pickle

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

def run():
    step = 0
    vehicle_pos = {} # Key: Vehicle ID, Value: 2D list of time and position pairs

    while traci.simulation.getMinExpectedNumber() > 0: # when vehicles are gone
        step += 1
        traci.simulationStep()

        # Retrieving positions of vehicles every 100 steps (10 seconds)
        if (step % 50) == 0:
            vehicleID = traci.vehicle.getIDList()
            #print(vehicleID)
            # After about 3 seconds
            #random_vehicle_ID = random.choice(vehicleID)
            #traci.vehicle.remove(vehicleID[0])
            for veh_id in vehicleID:
                if veh_id in vehicle_pos: # id already exists in the dictionary
                    vehicle_pos[veh_id].append([step//10, traci.vehicle.getPosition(veh_id)])
                else: # id doesn't exist in the dictionary
                    vehicle_pos[veh_id] = [[step//10, traci.vehicle.getPosition(veh_id)]]

        # Removing a vehicle
        # Need to make sure that we call getIDList function right before we remove the vehicle
        # ==> step should be divisible by 50 (or whatever number you chose)
        if sys.argv[1] == '1':
            if step == 700:
                random_vehicle_ID = random.choice(vehicleID)
                print(random_vehicle_ID)
                traci.vehicle.remove(random_vehicle_ID)
                #print(random_vehicle_ID)
                sys.stdout.flush()

    if sys.argv[1] == '1':
        remove_dict_val(vehicle_pos)
    #print(vehicle_pos)

    data_file = open('picklesub.pkl', 'wb')
    pickle.dump(vehicle_pos, data_file)
    data_file.close()

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "intersection1.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()