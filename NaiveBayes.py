import math
from Dataset import Dataset


class NaiveBayes(object):



	def __init__(self, trainset, testset):
		self.trainset = trainset
		self.testset  = testset

		# make a dict of dicts of dicts, where outermost keys are labels
		# outer keys are attrib names,
		# inner keys are attrib values, and vals are counts of occurrences
		self.xGivenYCounts = {lab:{attrib:{val:0 for val in trainset.attributeValues[attrib]} \
			for attrib in trainset.attributes} for lab in trainset.labels}
		self.yCounts = {lab:0 for lab in trainset.labels}


	def train(self):
		"""
		gets all of the counts for the various things
		"""
		# loop through instances
		for instance in self.trainset.instances:
			lab = instance[-1]
			self.yCounts[lab] += 1

			# loop through attributes
			for i in range(len(instance)-1):
				attrib = self.trainset.attributes[i]
				val = instance[i]
				self.xGivenYCounts[lab][attrib][val] += 1