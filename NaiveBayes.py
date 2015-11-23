import math
from Dataset import Dataset


class NaiveBayes(object):



	def __init__(self, trainset, testset, instList=None):
		self.trainset = trainset
		self.testset  = testset
		if instList:
			self.trainset.instances = instList

		# make a dict of dicts of dicts, where outermost keys are labels
		# outer keys are attrib names,
		# inner keys are attrib values, and vals are counts of occurrences
		self.xGivenYCounts = {lab:{attrib:{val:0 for val in trainset.attributeValues[attrib]} \
			for attrib in trainset.attributes} for lab in trainset.labels}
		self.yCounts = {lab:0 for lab in trainset.labels}




	def train(self, trainset):
		"""
		gets all of the counts for the various things
		"""
		# loop through instances
		#for instance in self.trainset.instances:
		for instance in trainset:
			lab = instance[-1]
			self.yCounts[lab] += 1

			# loop through attributes
			for i in range(len(instance)-1):
				attrib = self.trainset.attributes[i]
				val = instance[i]
				self.xGivenYCounts[lab][attrib][val] += 1



	def classify(self, instList):
		"""
		returns a list of tuples:
		the predicted class and its probability
		"""

		instClasses = []
		for inst in instList:

			ps_y_given_x = {}
			# need to calculate a probability associated with each label
			for lab in self.trainset.labels:

				numer = float((self.yCounts[lab]+1)) / \
				(sum(self.yCounts.values()) + len(self.yCounts.values()))
				
				for i in range(len(inst)-1):
					attrib = self.trainset.attributes[i]
					numer *= self.getPofX(attrib, inst[i], lab)

				denom = 0
				for lab1 in self.trainset.labels:
					tmpDenom = float((self.yCounts[lab1]+1)) / \
					(sum(self.yCounts.values()) + len(self.yCounts.values()))
					
					for i in range(len(inst)-1):
						attrib = self.trainset.attributes[i]
						tmpDenom *= self.getPofX(attrib, inst[i], lab1)

					denom += tmpDenom
				
				ps_y_given_x[lab] = numer/denom

			# get the max probability class
			maxClass = ''
			maxProb  = float("-inf")
			for key in self.trainset.labels:
				if ps_y_given_x[key] > maxProb:
					maxClass = key
					maxProb  = ps_y_given_x[key]

			tmp = [maxClass, maxProb]
			instClasses.append(tmp)

		return instClasses




	def getPofX(self, attrib, attribVal, lab):
		"""
		returns a laplace smoothed estimate of the probability of X = x
		given Y = y
		"""
		numerator = self.xGivenYCounts[lab][attrib][attribVal] + 1
		denominator = self.yCounts[lab] + len(self.trainset.attributeValues[attrib]) 
		return float(numerator)/denominator















