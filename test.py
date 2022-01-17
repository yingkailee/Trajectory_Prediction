import os
import sys
import optparse

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
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

# JUST EDIT THE CODE INSIDE THIS METHOD FOR MOST OF YOUR PURPOSES NOW
# SEE TRACI SUMO FOR MORE METHODS AND POSSIBLE TWEAKS TO YOUR SIMULATION
# TO USE YOUR OWN SIMULATION FILE, CHANGE 'test.sumocfg' AT THE BOTTOM OF
#   THIS FILE TO YOUR OWN '.sumocfg' FILE
def run():
    step = 0;
    while traci.simulation.getMinExpectedNumber() > 0: # when vehicles are gone
        step += 1
        traci.simulationStep()

        if step == 50:
             traci.vehicle.setSpeed("1", 0)

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "jacob2.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()
