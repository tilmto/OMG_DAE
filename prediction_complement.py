import csv  
import os
import sys

def prediction_complement(dir_name='./val_predict_raw'):
	files = os.listdir(dir_name)
	for file in files:
		with open(dir_name+'/'+file,'r') as rf:
			reader = csv.reader(rf)
			next(reader)

			with open('./val_predict/'+file,'w',newline='') as wf:
				writer = csv.writer(wf)
				writer.writerow(['valence'])
				writer.writerow([0])

				i = 1

				for item in reader:
					while int(item[0]) != i :
						writer.writerow([float(item[1])])
						i = i+1
					writer.writerow([float(item[1])])
					i = i+1

if __name__ == '__main__':
	prediction_complement('./val_predict_raw')
