import copy

class FSM:
	def __init__(self):
		self.transitionList = []
		self.numberOfStates = 0
		self.numberOfInputs = 0
		self.numberOfOutputs = 0
		self.transitionsNumber = 0
		self.initialState = 0

	def printTransitionTable(self):
		[print (i) for i in (self.transitionList)]

	def readFSMfromFile (self, fsm_filename):
		fsm_file = open(fsm_filename, 'r')
		fsm_file.readline()
		# head = FSM()
		line = fsm_file.readline()
		c_line = line.split()
		self.numberOfStates = int(c_line[1])
		line = fsm_file.readline()			#i 5
		c_line = line.split()
		self.numberOfInputs = int(c_line[1])	
		line = fsm_file.readline()			#o 10
		c_line = line.split()
		self.numberOfOutputs = int(c_line[1])
		line = fsm_file.readline()			# n0 0
		c_line = line.split()
		self.initialState = int(c_line[1])
		line = fsm_file.readline()		#p 10
		c_line = line.split()
		self.transitionsNumber = int(c_line[1])
		for i in range (self.transitionsNumber):
			line = fsm_file.readline()
			c_line = line.split()
			if len(c_line) != 4:
				print ("invalid FSM file")
				return
			self.transitionList.append(c_line)
		fsm_file.close()
		return self

	"""
	Создает массив States и массив Outputs
	"""
	def getStatesAndOutputsFromTest (self, inputs):
		outputs = []
		states = []
		currentState = self.initialState

		for input_ in inputs:
			states.append (currentState)
			if (input_ == -1):
				outputs.append(-1)
				states[-1] = -1
				currentState = self.initialState
				continue
			# print ("curs " + str(currentState) + " inp " + str(input_))
			for i in range (len(self.transitionList)):
				if (str(currentState) == str(self.transitionList[i][0]) and str(input_) == str(self.transitionList[i][1])):
					outputs.append(self.transitionList[i][3])
					currentState = self.transitionList[i][2]
					# print ("find " + str(i))
					break
				if (i == len(self.transitionList) - 1):
					print("Error! Can't find a transition " + str(currentState) + " " + input_)
		# print ("states")
		# print (states)
		states = [str(i) if i != -1 else i for i in states]
		return (states, outputs)


	'''
	Формирует множество достижимости для каждого состояния из последовательностей теста
	inputs это тест в виде одной последовательности объединенных тестовых последовательностей, разделенных символом -1 (reset)
	Каждая последовательность рассматривается без последнего символа, поскольку в тестовой последовательности должна 
	еще содержаться и различалка (ее минимальная длина 1)
	'''
	def getSCSet(self, inputs):
		#SCSet это список словарей для каждого состояния. ключ - длина, значение - множество (set)) посл-тей этой длины
		SCSet = [dict() for x in range(self.numberOfStates)]
		SCSet[0][0] = {''} # начальное состояние всегда достижимо по пустой последовательности
		# for q in SCSet:
		# 	print (q)
		currentState = self.initialState
		currentSeqStartingIndex = 0
		# print(inputs)
		for i in range (len(inputs)-1):
			if (inputs[i] == -1 or inputs[i+1] == -1):
				currentState = self.initialState
				currentSeqStartingIndex = i + 1
				continue
			curSeq = inputs[currentSeqStartingIndex:i+1]
			for transition in self.transitionList:
				if (str(currentState) == str(transition[0]) and str(inputs[i]) == str(transition[1])):
					if (str(transition[2]) != str(self.initialState)):	# для начального состояния не будем добавлять установки (хватит и пустой посл-ти)
						if (len(curSeq) in SCSet[int(transition[2])]): #если ключ (длина посл-тей) уже есть в словаре
							SCSet[int(transition[2])][len(curSeq)].add("".join(curSeq)) # то добавляем в множество связанное с этим ключем еще одну последовательность
						else:
							SCSet[int(transition[2])][len(curSeq)] = {"".join(curSeq)} # иначе создаем новую пару ключ-значение
					currentState = transition[2]
					break
			else:
				print("error!!! in getSCSet")

		return SCSet




	"""
	Проверка покрытия всех переходов тестом
	"""
	def checkTransitionsCovering(self, testFileName):
		transitionsMarks = []
		for i in range(self.transitionsNumber):
			transitionsMarks.append(0)			#отметки посещения переходов
		test_file = open (testFileName, 'r')
		for test_line in test_file:
			currentState = self.initialState
			test_line = test_line.split() #по пробелам
			for inp in test_line:
				if (inp.find('/') != -1):
					(inp, wq) = inp.split("/") #очередной входной символ
				for i in range(len(self.transitionList)):
					tr = self.transitionList[i]
					if (tr[0] == str(currentState) and tr[1] == str(inp)):
						transitionsMarks[i] = 1 #отметить посещенный переход
						currentState = tr[2]
						break
				else:
					print ("error while applying test to an FSM")
		test_file.close()
		if (transitionsMarks.count(1) != len (transitionsMarks)):
			return 0
		else:
			return 1




	"""
	Проверка достижения всех состояний тестом
	"""
	def checkStatesCovering(self, testFileName):
		statesMarks = []
		for i in range(self.numberOfStates):
			statesMarks.append(0)			#отметки посещения переходов
		test_file = open (testFileName, 'r')
		for test_line in test_file:
			currentState = self.initialState
			test_line = test_line.split() #по пробелам
			for inp in test_line:
				if (inp.find('/') != -1):
					(inp, wq) = inp.split("/") #очередной входной символ
				for i in range(len(self.transitionList)):
					tr = self.transitionList[i]
					if (tr[0] == str(currentState) and tr[1] == str(inp)):
						statesMarks[int(currentState)] = 1 #отметить посещенное состояние
						currentState = tr[2]
						break
				else:
					print ("error while applying test to an FSM")
		test_file.close()
		if (statesMarks.count(1) != len (statesMarks)):
			print("test doesn't cover all states")
			print (statesMarks)
		else:
			print ("test covers all states")


	"""
	делает из полного W теста неполный, удаляя все последовательности которыми достигается одно из состояний
	"""
	def makeNotCompleteWTestFromComplete (self, testFullName, testFullNameNotComplete):
		test_file = open(testFullName, 'r')
		test_file_new = open(testFullNameNotComplete, 'w')
		stateToExclude = int(self.numberOfStates) - 1
		numline = 0
		for line in test_file: 
			numline += 1
			foundStateToExclude = 0
			currentState = self.initialState
			splittedline = line.split()
			# print ("test line ", end='')
			# print (splittedline )
			for inp in splittedline:
				for i in range(self.transitionsNumber):
					if ((str(inp) == str(self.transitionList[i][1])) and (str(currentState) == str(self.transitionList[i][0]))):
						currentState = self.transitionList[i][2]
						# print (inp + "->" + currentState + ' ', end='')
						if (str(currentState) == str(stateToExclude)):
							foundStateToExclude = 1
							# print (" deleting (in line) " + str(numline))
							break
						break
				if (foundStateToExclude == 1):
					break
			# print ("leaving a sequence that give a trace:")
			# currentState = self.initialState
			# print (splittedline)
			# for ii in splittedline:
			# 	for i in range(self.transitionsNumber):
			# 		if ((str(ii) == str(self.transitionList[i][1])) and (str(currentState) == str(self.transitionList[i][0]))):
			# 			currentState = self.transitionList[i][2]
			# 			print (ii + "->" + currentState + ' ', end='')	
			if (foundStateToExclude == 1):
				continue
			# print ()

			line = " ".join(splittedline)

			test_file_new.write(line + '\n')
		test_file.close()
		test_file_new.close()

