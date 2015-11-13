"""
Trains Naive Bayes or Tree Augmented Naive Bayes

Takes a training arff file, a test set arff file,
and either "n" (vanilla naive bayes) or "t"
(tree augmented naive bayes)

Rick Wolf
"""

import sys
import math
from Dataset import Dataset
from NaiveBayes import NaiveBayes




def readFile(fname):
		"""
		Takes a filename of an ARFF file from the current directory and returns
		a Dataset object
		"""

		attributes = [] # an ordered list of the attribute names
		attributeValues = {} # a dictionary with attrib name as keys
		instances = []
		labels = []

		lines = open(fname, 'r')
		for line in lines:
			if not (line.startswith("%")):
				line = line.strip("\n")
				if (line.startswith("@attribute")):
					# line is now a list of words
					line = [word.lstrip("{ '\t").rstrip("} ',\t") for word in line.split() if word != '{']
					if (line[1].lower() == "class"):
						labels = line[2:]
						labels = [x for x in labels if x] # removes any empty strings
					else:
						attributes.append(line[1]) # the name will always be the second item
						attributeValues[line[1]] = line[2:] #the third to end of the list is the values
				elif  not (line.startswith("@")):
					line = line.split(',')
					newline = []
					for i in range(len(attributes)):
						if len(attributeValues[attributes[i]]) == 1:
							newline.append(float(line[i]))
						else:
							newline.append(line[i])
					newline.append(line[-1])
					instances.append(newline[:])
					
		return Dataset(labels, attributes, attributeValues, instances)


def main(argv):

	trainfile = ''
	testfile  = ''
	mode      = ''

	# validate input
	if len(sys.argv) == 4:
		trainfile = sys.argv[1]
		testfile  = sys.argv[2]
		mode      = sys.argv[3]
	else:
		print("incorrect input supplied")
		sys.exit()


	# ingest data
	trainset = readFile(trainfile)
	testset  = readFile(testfile)

	y1 = 0
	y2 = 0
	for instance in trainset.instances:
		if instance[-1] == trainset.labels[0]:
			y1 +=1
		else:
			y2 +=1
	#print trainset.labels
	#print trainset.attributeValues
	print trainset.labels[0], y1
	print trainset.labels[1], y2

	bayes = NaiveBayes(trainset, testset)
	bayes.train()
	print bayes.yCounts
	print bayes.xGivenYCounts[trainset.labels[0]]['bl_of_lymph_c'].values()
	print bayes.xGivenYCounts[trainset.labels[1]]['bl_of_lymph_c'].values()


if __name__ == '__main__':
	main(sys.argv)










