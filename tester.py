import pickle

if __name__ == "__main__":
	data_file = open('pickle.pkl', 'rb')
	while 1:
		try:
			x = pickle.load(data_file)
			print(x)
		except EOFError:
			break