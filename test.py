import os
import sys
import random
import pickle

def append_files(data_file, file1):
	file = open(file1, 'rb')
	pickle.dump(pickle.load(file), data_file)

def main():
	lowerBound = 80
	upperBound = 100

	"""
	# Create one file for different X (pickle file)
	# Store the removed vehicle's ID as well
	# Change the map
	# Function to read info from pickle file
	Have one pickle file in the end
	# of X
	Y_gc
	Y0
	Y1
	Y_gc
	Y0
	Y1
	""" 

	X = 2 # Number of simulations with different initial conditions (1000)
	Y = 3 # Number of times we remove one random vehicle from the same simulation (one additional ground truth)
	cmd_seahorse = 'python seahorse.py '
	network = ' -n intersection1.net.xml'
	route = ' -r intersection1.rou.xml'

	data_file = open('pickle.pkl','wb')

	for i in range(X):
		n = random.randint(lowerBound, upperBound)
		commandy = 'python $HOME/Downloads/sumo/tools/randomTrips.py'
		commandy += network + route + ' -e'
		commandy += str(n)
		commandy += ' --random'
		os.system(commandy)
		print("Number:", i)
		for j in range(Y):
			if j == 0:
				os.system(cmd_seahorse + '0')
			else:
				os.system(cmd_seahorse + '1')
			append_files(data_file, 'picklesub.pkl')

	
	file = open('pickle.pkl', 'rb')
	count = 0
	try:
		while True:
			obj = pickle.load(file)
			print(obj)
			count = count + 1
	except EOFError:
		print('End of File')
		print(count)
	file.close()



if __name__ == "__main__":
	main()
