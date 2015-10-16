import pickle

def p_save(obj, filename):
    with open("processed_data/" + filename, 'wb') as outfile:
        pickle.dump(obj, outfile, 2)
		
def p_load(filename):
    with open("processed_data/" + filename, 'rb') as infile:
        return pickle.load(infile)