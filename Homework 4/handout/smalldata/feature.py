import sys
import csv

train_input = sys.argv[1]
valid_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
format_train = sys.argv[5]
format_valid = sys.argv[6]
format_test = sys.argv[7]
feature_flag = int(sys.argv[8])

def format(input_file, output_file, model_number):
	words = []
	with open(input_file) as f:
		tsvreader = csv.reader(f, delimiter = '\t')
		c = 0
		for line in tsvreader:
			words.append(line[1].split(' '))
			words[c].insert(0, line[0])
			c += 1

	features = []

	for i in range(len(words)):
		features.append([])
		features[i].append(words[i][0])
		existing = []
		for j in range(1, len(words[i])):
			if words[i][j] in vocab and words[i][j] not in existing:
				if model_number == 1:
					features[i].append(vocab[words[i][j]] + ':' + '1')
				else:
					if words[i].count(words[i][j]) < t:
						features[i].append(vocab[words[i][j]] + ':' + '1')
				existing.append(words[i][j])


	with open(output_file, 'w') as f:
		tsv_writer = csv.writer(f, delimiter = '\t')
		for i in range(len(features)):
			tsv_writer.writerow(features[i])


vocab = {}
t = 4 #threshold

#read vocabulary and store in dictionary vocab
with open(dict_input) as f:
	for line in f:
		(key, val) = line.split()
		vocab[key] = val

format(train_input, format_train, feature_flag)
format(valid_input, format_valid, feature_flag)
format(test_input, format_test, feature_flag)