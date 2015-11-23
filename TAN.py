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
		self.train(trainset.instances)
		



	def initializeGraph(self):
		"""
		returns a list of tuples that consist of:
		(Xi, Xj, I(Xi,Xj|Y))
		"""
		edges = []
		# Xi loop
		for XI in self.trainset.attributes:
			XIind = self.trainset.attributes.index(XI)
			XIlen = len(self.trainset.attributeValues[XI])
			#Xj loop
			for XJ in self.trainset.attributes:
				XJind = self.trainset.attributes.index(XJ)
				XJlen = len(self.trainset.attributeValues[XJ])

				if XI != XJ:
					condMutInf = 0
					# get joint prob P(xi,xj|y)
					for xi in self.trainset.attributeValues[XI]:
						for xj in self.trainset.attributeValues[XJ]:
							for lab in self.trainset.labels:							
								jpCount = 0
								for inst in self.trainset.instances:
									if (inst[-1] == lab) and (xi == inst[XIind]) and (xj == inst[XJind]):
										jpCount += 1
								
								jp  = float(jpCount + 1) / (len(self.trainset.instances) + \
									(len(self.trainset.labels)*XIlen*XJlen))
								jcp = float(jpCount + 1) / (self.yCounts[lab] + (XIlen*XJlen))
								
								pXi = self.getPofX(XI,xi,lab)
								pXj = self.getPofX(XJ,xj,lab)
								condMutInf += jp * math.log(jcp/(pXi*pXj),2)

					edges.append((XI,XJ,condMutInf))
		return edges


	def growPrim(self, edges):
		"""
		Uses Prim's algorithm to grow the maximum spanning tree
		"""
		v = self.trainset.attributes
		vNew = [self.trainset.attributes[0]]
		eNew = []

		while set(v) != set(vNew):

			maxInd = 0
			maxInf = .00000000001
			for i in range(len(edges)):
				edge = edges[i]
				if (edge[0] in vNew) and (edge[1] not in vNew):
					if math.log(edge[2]) > math.log(maxInf):
						maxInf = edge[2]
						maxInd = i
			vNew.append(edges[maxInd][1])
			eNew.append(edges[maxInd])

		return (vNew, eNew)



	def setParentList(self, edges):
		"""
		returns a dict with attributes as keys and parents of those attributes
		as values
		"""
		self.parentList = {}

		for attrib in self.trainset.attributes:
			self.parentList[attrib] = []

			for edge in edges:
				if edge[1] == attrib:
					self.parentList[attrib].append(edge[0])
			#self.parentList[attrib].append('class')



	def classify(self, instList):

		instClasses = []
		for inst in instList:
			ps_y_given_x = {}
			# need to calculate a probability associated with each label
			for lab in self.trainset.labels:

				numer = float((self.yCounts[lab]+1)) / \
				(sum(self.yCounts.values()) + len(self.yCounts.values()))
				
				for i in range(len(inst)-1):
					attrib = self.trainset.attributes[i]
					numer *= self.getProbXgivenParents(attrib, inst[i], lab, inst)

				denom = 0
				for lab1 in self.trainset.labels:
					tmpDenom = float((self.yCounts[lab1]+1)) / \
					(sum(self.yCounts.values()) + len(self.yCounts.values()))
					
					for i in range(len(inst)-1):
						attrib = self.trainset.attributes[i]
						tmpDenom *= self.getProbXgivenParents(attrib, inst[i], lab1, inst)

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



	def getProbXgivenParents(self, attrib, attribVal, lab, inst):

		p = 1
		parents = self.parentList[attrib]
		attribInd = self.trainset.attributes.index(attrib)

		if parents:
			parentInd = self.trainset.attributes.index(parents[0])
			xi = inst[parentInd]

			jpCount = 0
			parentsCount = 0
			for inst in self.trainset.instances:
				if (inst[attribInd] == attribVal) and (inst[parentInd] == xi) and (inst[-1] == lab):
					jpCount += 1
					parentsCount += 1
				elif (inst[parentInd] == xi) and (inst[-1] == lab):
					parentsCount += 1
			jpc = float(jpCount+1)/(parentsCount + \
				(len(self.trainset.attributeValues[attrib]) ))
			p *= jpc
		else:
			p = self.getPofX(attrib, attribVal, lab)

		return p










		