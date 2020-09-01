#!/usr/bin/python

import sys
import os
import subprocess
import time
from datetime import datetime
sys.path.append(".")
from FSM import *
from statistics import variance, mean
from makeFormula import *
from itertools import product
import math


# прописать путь к директории с проектом экспериментов и исполняемым файлам вспомогательных утилит
workingDir = "/home/andrey/z3/bin/python/" # тут хранятся все вспомогательные директории со скриптами и генерируемыми файлами
testType = "TT" # W or TT or H
testexp = "notcomplete" # complete or notcomplete
pathToMakeFormula = workingDir + "progs/makeFormula.py"
# pathToFSMGenerator = "/home/andrey/exp1901_3/bin/gen_fsm"
pathToFSMGenerator = "/../path/to/gen_fsm"



"""
делает из полного ТТ теста неполный, удаляя один последний символ из первой последовательности
"""
def makeNotCompleteTestFromComplete (testFullName, testFullNameNotComplete):
	test_file = open(testFullName, 'r')
	test_file_new = open(testFullNameNotComplete, 'w')
	counter = 1
	for line in test_file: 
		if (counter == 1):
			splittedline = line.split()
			splittedline.pop(-1)
			line = " ".join(splittedline)
		test_file_new.write(line + '\n')
		counter += 1
	test_file.close()
	test_file_new.close()

# k - количество столбцов в формируемой матрице
def makeMatrixFromList(a, k):
	matrix = [[None for i in range(k)] for x in range (len(a)//k )]
	if (len(a) % k):
		matrix.append([None for i in range(k)])
	# print (len(matrix))
	for i in range (0, len(a)):
		matrix[i//k][i%k] = a[i]
	return matrix

def generateFSM(workingDir, fsmFullName, statesNum, inputsNum, outputsNum):

	time.sleep(1)	# надо ждать чтобы генерировались разные автоматы (т.к. криво работает rand в генераторе автоматов)
	out = 1
	gencounter = 50
	while out != 0 and gencounter > 0:
		out = os.system(pathToFSMGenerator + " -R " + str(statesNum) + ' ' + str(inputsNum) + ' ' + str(outputsNum) + ' 1 ' + workingDir + "FSMs/")
		gencounter -= 1
	time.sleep(1)
	if (out != 0):
		print ("cant generate one FSM " + fsmFullName)
		# print (out)
		return 1
	os.replace (workingDir + "FSMs/0.fsm", fsmFullName)
	if (os.path.exists(fsmFullName) == 0):
		print ("error! no FSM")
		return 1
	return 0

def generateTest(fsmFullName, testFullName, testType):
	if (testType == "TT"):
		pathToTestGenerator = "/......path/to/transitionTour"
	elif (testType == "W" or testType == "H"):
		pathToTestGenerator = "wine /...path/to/test_fsm.exe"
	if (testType == "TT"):
		os.system (pathToTestGenerator + ' ' + fsmFullName + ' ' + testFullName)
	elif (testType == "W"):
		os.system (pathToTestGenerator + ' ' + fsmFullName + ' ' + testFullName + " 1 " + '0')
	elif (testType == "H"):
		os.system (pathToTestGenerator + ' ' + fsmFullName + ' ' + testFullName + " 3 " + '0')
	if (os.path.exists(testFullName) == 0 or os.path.getsize(testFullName) == 0):
		print ("error!!! have no test or empty test")
		return 1
	return 0

def makeAnotherTest(numberOfStates, numberOfInputs):
	fsmname = "/home/andrey/z3/bin/python/otherFSMs/Uncomplete.fsm"
	testFullName = "/home/andrey/z3/bin/python/otherFSMs/Uncomplete.fsm.test"
	k = generateFSM("/home/andrey/z3/bin/python/", fsmname , numberOfStates, numberOfInputs, numberOfInputs)
	if (k != 0):
		print("error! " + str(k) )
		print ("pause")
		pause_variable = input()
	k1 = generateTest(fsmname, testFullName, "TT")
	if (k1 != 0):
		print("error2!!!!")
		print ("pause")
		pause_variable = input()
	fsm = FSM()
	fsm.readFSMfromFile(fsmname)
	print ("made another test")
	return (fsm, testFullName)

def getItListFromSC(SC):
	SClist = [[] for x in range (fsm.numberOfStates)] #список списков, где для каждого состояния - просто список последовательностей
	for i in range(len(SC)):
		for key in SC[i]:
			for k in SC[i][key]:
				SClist[i].append(k)
	# print ("SClist:")
	# for i in SClist:
	# 	print(i)
	sequencesNum = [0 for i in range(fsm.numberOfStates)] # количество уст. последовательностей из SC для каждого состояния
	for i in range(len(SC)):
		for key in SC[i]:
			for k in SC[i][key]:
				sequencesNum[i] += 1
	sequencesNumLst = [[] for i in range (len(sequencesNum))] # для каждого состояния список (1,2,..n) n - это число уст. посл-й состояния
	for i in range (len(sequencesNum)):
		for j in range(sequencesNum[i]):
			sequencesNumLst[i].append(j)
	# print (sequencesNumLst)
	sequencesNumMatrix = [[] for i in range (len(sequencesNum))] # для каждого состояния список списков ([1,2],[3,4]..[n-1,n]) n - это число уст. посл-й состояния
	for i in range (len(sequencesNumMatrix)):
		sequencesNumMatrix[i] = makeMatrixFromList(sequencesNumLst[i], 1)

	for i in sequencesNumMatrix:
		print (i)
	itlist = list(product (*sequencesNumMatrix)) # список кортежей декартова произведения
	return (SClist, itlist) 





now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")

logFileName = workingDir + 'logs/' + 'log' + dt_string
logFile = open(logFileName, 'w')
logFile.write (testType + ' ' + testexp + "\n")

fsmsNum = 10
if (fsmsNum < 2 and testType == 'TT'):
	print("can't do experiments with only one FSM")
for statesNum in range (4, 9, 2):
	for inputsNum in range (2, 9, 2):
		FSMname = "s" + str(statesNum) + "i" + str(inputsNum)
		testName = ""
		for fsmNum in range (fsmsNum): #гненерация автоматов и тестов
			fsmFullName = workingDir + "FSMs" + testType + testexp + "/" + FSMname + "_" + str (fsmNum) + ".fsm"
			if (generateFSM(workingDir, fsmFullName, statesNum, inputsNum, inputsNum) != 0):
				continue
			testFullName = fsmFullName + ".test" + testType
			if (generateTest(fsmFullName, testFullName, testType) != 0):
				continue
		fsmNum = 0
		errorsCounter = 0
		logFile = open(logFileName, 'a')
		timeValues = []
		for fsmNum in range (fsmsNum): # генерация и проверка формул
			fsmFullName = workingDir + "FSMs" + testType + testexp + "/" + FSMname + "_" + str (fsmNum) + ".fsm"
			testFullName = fsmFullName + ".test" + testType
			# fsmFullName = "/home/andrey/z3/bin/python/FSMsHcomplete/s5i2_2.fsm"
			# testFullName = "/home/andrey/z3/bin/python/FSMsHcomplete/s5i2_0.fsm.testH"

			print ("start " + fsmFullName )
			print (testFullName)
			logFile.write("start " + fsmFullName + '\n')
			logFile.write(testFullName + '\n')

			fsm = FSM()
			fsm.readFSMfromFile (fsmFullName) 

			formulaFileName = workingDir + "formula.py"
			formulaNewFileName = workingDir + "FSMs" + testType + testexp + "/" + FSMname + "_" + str (fsmNum) + ".py"
			if (testexp == "notcomplete"):
				testFullNameNotComplete = testFullName + "not_complete"
				if (testType == "W" or testType == "H"):
					# fsm.makeNotCompleteWTestFromComplete(testFullName, testFullNameNotComplete)
					# #fsm.checkTransitionsCovering(testFullNameNotComplete)
					# #fsm.checkStatesCovering(testFullNameNotComplete)
					pass
				elif (testType == 'TT'):
					# makeNotCompleteTestFromComplete (testFullName, testFullNameNotComplete)
					cond=1
					q=1
					while (cond==1 and q < fsmsNum):
						testFullNameNotComplete = workingDir + "FSMs" + testType + testexp + "/" + FSMname + "_" + str ((fsmNum + q) % fsmsNum) + ".fsm" + ".test" + testType
						cond = fsm.checkTransitionsCovering(testFullNameNotComplete)
						q+=1
					print (str(q) + " case worked")
					testFullName = testFullNameNotComplete

			if (testType == 'H'):
				# completecheck = 1
				# while (completecheck == 1):
				# 	print("trying again")
				# 	logFile.write("trying again\n")
				# 	completecheck = 0
				# 	(fsm,testFullName) = makeAnotherTest(fsm.numberOfStates, fsm.numberOfInputs)
				(inputs, outputs) = readTestFromFile(testFullName)
				(states, outputs) = fsm.getStatesAndOutputsFromTest(inputs)
				SC = fsm.getSCSet(inputs)
				for i in SC:
					print (i)

				(SClist, itlist) = getItListFromSC(SC)
				# print (itlist)

				itnum = len (itlist)
				print ("всего %d комбинаций" % itnum)
				output = ''
				i = 0
				alltime = 0
				# while (str(output) != "b'sat\\n'" and i < itnum):
				while (i < itnum): # проверка неполного теста, нужно обязательно все итерации пройти
					print ('комбинация %d: ' % i)
					logFile.write('комбинация %d: \n' % i)
					tempSCset = [dict() for x in range(fsm.numberOfStates)]
					for s in range (fsm.numberOfStates):
						for n in itlist[i][s]: #n это список 
							if n != None:
								if len(SClist[s][n]) not in tempSCset[s]:
									tempSCset[s][len(SClist[s][n])] = set({SClist[s][n]})
								else:
									tempSCset[s][len(SClist[s][n])].add(SClist[s][n])
					# for j in tempSCset:
					# 	print (j)
					logFile.write(str(tempSCset) + '\n')
					start_time = time.time()
					makeFormulaForH(formulaFileName, fsm, inputs, outputs, states, tempSCset)
					output = subprocess.check_output("python3 " + formulaFileName, shell=True)
					end_time = time.time() - start_time
					print (end_time)
					alltime += end_time
					logFile.write('%s\n' % end_time)
					logFile.flush()
					if (str(output) == "b'sat\\n'"):
						print("Error!!!! it is SAT")
						print ("pause")
						pause_variable = input()
						completecheck = 1
						break
					i += 1
				print("time for all iterations is %s" % alltime)
				logFile.write("time for all iterations is %s\n" % alltime)
				logFile.flush()
			else:
				start_time = time.time()
				os.system ("python3 " + pathToMakeFormula + ' ' + fsmFullName + ' ' + testFullName + ' ' + formulaFileName + ' ' + testType)
				output = subprocess.check_output("python3 " + formulaFileName, shell=True)
				end_time = time.time() - start_time

			if ((testexp=="complete" and str(output) != "b'sat\\n'") or (testexp=="notcomplete" and str(output) != "b'unsat\\n'")):
				print ("error in " + formulaNewFileName + "\n" + str(output))
				logFile.write ("error in " + formulaNewFileName + "\n" + str(output) + "\n")
				logFile.flush()
				errorsCounter +=1
			else:
				timeValues.append(end_time)
			# os.replace (formulaFileName, formulaNewFileName)
			print(fsmFullName + "_" + str(fsmNum) + " " +  str (end_time))
			if (testType == "H"):
				logFile.write (fsmFullName + "_" + str(fsmNum) + " " + str (end_time) + " " + str (i) + " iteration\n")
			else:
				logFile.write (fsmFullName + "_" + str(fsmNum) + " " + str (end_time) + '\n')
			logFile.flush()
		avgTime = mean (timeValues)
		stdev = math.sqrt(variance(timeValues))
		if (errorsCounter > 0):
			print(" Warning!!! There was %d errors" % errorsCounter)
		logFile.write ("%d_%d_avg_time = %s +- %s\n" % (statesNum, inputsNum, avgTime, stdev))	
		logFile.close ()





