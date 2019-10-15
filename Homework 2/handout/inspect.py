import sys
import csv
import math

numb_of_label_1 = 0
numb_of_label_2 = 0
error_rate = 0

#read tsv file
with open(sys.argv[1]) as tsv_file:
	tsvreader = csv.reader(tsv_file, delimiter = '\t')
	tsvreader.__next__()
	label_1 = tsvreader.__next__()[-1]
	numb_of_label_1 += 1

	#count number of positive/negative labels
	for line in tsvreader:
		if line[-1] == label_1:
			numb_of_label_1 += 1
		else:
			numb_of_label_2 += 1

total_examples = numb_of_label_1 + numb_of_label_2

#calculate probabilities and entropy
p1 = numb_of_label_1 / total_examples
p2 = numb_of_label_2 / total_examples

entropy = - p1 * math.log2(p1) - p2 * math.log2(p2)

#calculate error rate based on which label prevails, CHECK ERROR RATE AGAIN!
if numb_of_label_1 >= numb_of_label_2:
	error_rate = p2
else:
	error_rate = p1

#write required values to file
with open(sys.argv[2], 'w') as output_file:
	output_file.write(f'entropy: {entropy}\nerror: {error_rate}')