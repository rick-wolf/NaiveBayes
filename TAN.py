import math
from Dataset import Dataset
from NaiveBayes import NaiveBayes


class TAN(NaiveBayes):
	"""
	use's Prim's algorithm to find a Tree Structure for Naive Bayes
	This class inherits from the Naive Bayes Class
	"""

	def __init__(self, trainset, testset):
		NaiveBayes.__init__(self, trainset, testset)
		self.train()



	def initializeGraph(self):
		"""
		returns a list of tuples that consist of:
		(Xi, Xj, I(Xi,Xj|Y))
		"""
		vertices = []

		
		# Xi loop
		for XI in self.trainset.attributes:
			#Xj loop
			for XJ in self.trainset.attributes:

				condMutInf = 0
				# label loop
				for lab in self.trainset.labels:
					
					print lab, self.yCounts[lab]
					# get joint prob P(xi,xj|y)
					for xi in self.trainset.attributeValues[XI]:
						for xj in self.trainset.attributeValues[XJ]:							
							jpCount = 0
							for inst in self.trainset.instances:
								if (inst[-1] == lab) and (xi in inst) and (xj in inst):
									jpCount += 1
							
							jp  = float(jpCount + 1) / (len(self.trainset.instances) + 1)
							jcp = jp / (float(self.yCounts[lab] + 1)/(len(self.trainset.instances) + 1))

							pXi = self.getPofX(XI,xi,lab)
							pXj = self.getPofX(XJ,xj,lab)
							condMutInf += jp * math.log(jcp/(pXi*pXj),2)
							print jp, (float(self.yCounts[lab] + 1)/(len(self.trainset.instances) + 1)), pXi, pXj, condMutInf

					vertices.append((XI,XJ,condMutInf))
		return vertices








		