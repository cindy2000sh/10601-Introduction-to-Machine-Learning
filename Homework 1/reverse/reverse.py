import sys

#Open input file and save lines into a list
with open(sys.argv[1], 'r') as input_file:
	text = input_file.readlines()
reverse_text = []

#Store the lines in reversed order in a new list
for i in reversed(text):
	reverse_text.append(i)

#Write the reverse order lines to an output file
with open(sys.argv[2], 'w') as output_file:
	for i in reverse_text:
		output_file.write(i)