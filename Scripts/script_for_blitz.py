import numpy as np

def get_name():
	name_u = "u.txt" 
	name_v = "v.txt" 
	name_p = "p.txt" 


def get_data(filepath):
	file = open(filepath)
	lines = file.readlines()
	lines = [line.rstrip() for line in lines]
	lines = lines[1:]
	u = list()

	for line in lines:
	    row = line.split(" ")
	    row_entries = list()
	    for element in row:
	        try:
	            row_entries.append(float(element))
	        except:
	            pass
	    u.append(row_entries)
	return np.array(u)

def get_all_data():
	name_u, name_v, name_p = get_name()
	u, v, p = get_data(name_u), get_data(name_v), get_data(name_p)
	return u, v, p

if __name__=="__main__":
	u,v,p = get_all_data()