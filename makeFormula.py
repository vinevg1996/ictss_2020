#!/usr/local/bin/python

# 1) Считать автомат из файла
# 2) Считать тест из файла
# 3) Построить на основе автомата и теста массив State и Output
# 4) сформировать файл с исходным текстом программы для проверки формулы
# 

import sys
import argparse
sys.path.append(".")
from FSM import FSM

"""
Считывает из файла тест (как в формате i1 i2 i3... так и в формате i1/o1 i2/o2...)
Возвращает массив inputs входных символов теста (одной последовательностью) и массив outputs реакций
(cимволом -1 обозначается reset и реакция на него)
"""
def readTestFromFile(test_filename):
	inputs = []
	outputs = []
	test_file = open(test_filename, 'r')
	line = test_file.readline()
	test_file.seek(0)
	if (line.find('/') != -1): # cчитываем тест в формате i/o 
		for line in test_file: 
			for it in line.split():
				(i, o) = it.split('/')
				inputs.append(i)
				outputs.append(o)
			inputs.append (-1)
			outputs.append (-1)
			# for i in range(len(inputs)):
			# 	print (str(inputs[i]) + "/" + str(outputs[i]) + " ", end = "")
	else: #cчитываем тест в формате i i i 
		for line in test_file: 
			# print ("test line " + line)

			inputs += line.split()
			inputs.append (-1)
	inputs.pop(-1)
	if (outputs):
		outputs.pop(-1)
	return (inputs, outputs)

# разборщик параметров
def createParamParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('fsmFile', type = str, action = 'store', help = 'path to the file with an FSM in KITIDIS format')
    parser.add_argument('testFile', type = str, action = 'store', help = 'path to the file with a test for a given FSM')
    parser.add_argument('mainFile', type = str, action = 'store', help = 'path to file with the created formula')
    parser.add_argument('test_type', type = str, action = 'store', help = 'TT or W or H')
    return parser


def makeFormulaForTT (mainFileName, fsm, inputs, states):
	while -1 in inputs : inputs.remove(-1)
	while -1 in outputs : outputs.remove(-1)
	while -1 in states : states.remove(-1)
	# формируем файл .py с формулой
	mainFile = open(mainFileName, 'w')
	mainFile.write ("#!/usr/bin/python\nfrom z3 import *\n\n")
	mainFile.write ("Test = Array('Test', IntSort(), IntSort())\n")
	# for i in range (len(inputs)):		# !!! переделать в виде цикла
	# 	mainFile.write("Test = Store(Test, " + str(i) + ", " + str(inputs[i]) + ")\n")

	mainFile.write ("Test_seq = [")
	for i in range (len (inputs) -1):
		mainFile.write(str(inputs[i]) + ", ")
	mainFile.write(str(inputs[-1]) + "]\n")

	mainFile.write ("\nlen_test = len(Test_seq)\n")
	mainFile.write ("for i in range(0, len_test):\n")
	mainFile.write ("\tTest = Store(Test, i, Test_seq[i])\n")

	mainFile.write("\nState = Array('State', IntSort(), IntSort())\n")
	# for i in range (len(states)):
	# 	mainFile.write("State = Store(State, " + str(i) + ", " + str(states[i]) + ")\n")

	mainFile.write ("State_seq = [")
	for i in range (len (states) -1):
		mainFile.write(str(states[i]) + ", ")
	mainFile.write(str(states[-1]) + "]\n")

	mainFile.write ("for i in range(0, len_test):\n")
	mainFile.write ("\tState = Store(State, i, State_seq[i])\n")

	mainFile.write ("\nsolver = Solver()\ns, i = Ints('s i')\nj = Int('j')\n")
	mainFile.write ("Constr_s_i = And(s >= 0, s < " + str(fsm.numberOfStates) + ", i >= 0, i < " + str(fsm.numberOfInputs) + ")\n");
	mainFile.write ("f1 = Exists([j], And(j >= 0, j < " + str(len(inputs)) + ", s == State[j], i == Test[j]))\n")
	mainFile.write ("F = ForAll([s, i], Implies(Constr_s_i, f1))\n")
	mainFile.write ("solver.add(F)\nsolver_check = solver.check()\nprint (solver_check)\n")
	mainFile.close()


def makeFormulaForW (mainFileName, fsm, inputs, outputs, states):
	inputs = [fsm.numberOfInputs if x==-1 else x for x in inputs]
	states = [fsm.numberOfStates if x==-1 else x for x in states]
	outputs = [fsm.numberOfOutputs if x==-1 else x for x in outputs]
	mainFile = open(mainFileName, 'w')
	mainFile.write ("#!/usr/bin/python\nfrom z3 import *\n\n")
	mainFile.write ("def test_diff_s0_s1():\n")
	mainFile.write ("\tTest = Array('Test', IntSort(), IntSort())\n")
	mainFile.write ("\tState = Array('State', IntSort(), IntSort())\n")
	mainFile.write ("\tOut = Array('Out', IntSort(), IntSort())\n")
	mainFile.write ("\tTest_seq = [")
	for i in range (len(inputs) - 1):
		mainFile.write (str(inputs[i]) + ", ")
	mainFile.write (str(inputs[-1]) + "]\n")
	mainFile.write ("\tState_seq = [")
	for i in range (len(states) - 1):
		mainFile.write (str(states[i]) + ", ")
	mainFile.write (str(states[-1]) + "]\n")
	mainFile.write ("\tOut_seq = [")
	for i in range (len(outputs) - 1):
		mainFile.write (str(outputs[i]) + ", ")
	mainFile.write (str(outputs[-1]) + "]\n")	

	mainFile.write ("\n\tlen_test = len(Test_seq)\n")
	mainFile.write ("\tfor i in range(0, len_test):\n")
	mainFile.write ("\t\tTest = Store(Test, i, Test_seq[i])\n")
	mainFile.write ("\tfor i in range(0, len_test):\n")
	mainFile.write ("\t\tState = Store(State, i, State_seq[i])\n")
	mainFile.write ("\tfor i in range(0, len_test):\n")
	mainFile.write ("\t\tOut = Store(Out, i, Out_seq[i])\n")
	mainFile.write ("\treturn [Test, State, Out, len_test]\n")

	mainFile.write("\ndef check_for_state(s1, s2, solver, start_ids, x_vars, y_vars, State, Test, Out, inputs_number):\n")
	mainFile.write("\tj1 = start_ids[s1][s2]\n")
	mainFile.write("\tj2 = start_ids[s2][s1]\n")
	mainFile.write("\tstate_constr = And(State[j1] == s1, State[j2] == s2)\n")
	mainFile.write("\tsolver.add(state_constr)\n")
	mainFile.write("\ty = y_vars[s1][s2]\n")
	mainFile.write("\tw = ForAll([y],\n")
	mainFile.write("\t\t\tImplies(\n")
	mainFile.write("\t\t\t\tAnd(y >= 0, y < x_vars[s1][s2], Test[j1 + y] != inputs_number, Test[j2 + y] != inputs_number),\n")
	mainFile.write("\t\t\t\tAnd(Test[j1 + y] == Test[j2 + y], Out[j1 + y] == Out[j2 + y])))\n")
	mainFile.write("\tg = And(w, Test[j1 + x_vars[s1][s2]] == Test[j2 + x_vars[s1][s2]], Out[j1 + x_vars[s1][s2]] != Out[j2 + x_vars[s1][s2]])\n")
	mainFile.write("\tsolver.add(g)\n\treturn\n")

	mainFile.write("states_number = " + str(fsm.numberOfStates) +  "\ninputs_number = " + str(fsm.numberOfInputs) + "\n")

	mainFile.write("\nstart_ids = [[Int('start_ids_%d_%d' % (i, j)) for j in range(0, states_number) ]\n\t\t\tfor i in range(0, states_number) ]\n")
	mainFile.write("x_vars = [[Int('x_vars_%d_%d' % (i, j)) for j in range(0, states_number) ]\n\t\t\tfor i in range(0, states_number) ]\n")
	mainFile.write("y_vars = [[Int('y_vars_%d_%d' % (i, j)) for j in range(0, states_number) ]\n\t\t\tfor i in range(0, states_number) ]\n")

	mainFile.write("\nsolver = Solver()\n[Test, State, Out, len_test] = test_diff_s0_s1()\n")

	mainFile.write("for s1 in range(0, states_number):\n\tfor s2 in range(0, states_number):\n\t\tif (s1 < s2):\n\t\t\tsolver.add(start_ids[s1][s2] >= 0, start_ids[s1][s2] < len_test)\n")
	mainFile.write("\t\t\tsolver.add(Test[start_ids[s1][s2]] != inputs_number)\n")
	mainFile.write("\t\t\tsolver.add(start_ids[s2][s1] >= 0, start_ids[s2][s1] < len_test)\n")
	            
	mainFile.write("\t\t\tsolver.add(Test[start_ids[s2][s1]] != inputs_number)\n")
	mainFile.write("\t\t\tsolver.add(x_vars[s1][s2] >= 0, x_vars[s1][s2] < states_number - 1)\n")
	mainFile.write("\t\t\tsolver.add(start_ids[s1][s2] + x_vars[s1][s2] < len_test, \n")
	mainFile.write("\t\t\t\t\t\tstart_ids[s2][s1] + x_vars[s1][s2] < len_test)\n")
	mainFile.write("\t\t\tsolver.add(Test[start_ids[s1][s2] + x_vars[s1][s2]] != inputs_number)\n")
	mainFile.write("\t\t\tsolver.add(Test[start_ids[s2][s1] + x_vars[s1][s2]] != inputs_number)\n")
	           
	mainFile.write("flag = 1\n")
	mainFile.write("for s1 in range(0, states_number):\n\tfor s2 in range(0, states_number):\n\t\tif (s1 < s2):\n")
	mainFile.write("\t\t\tcheck_for_state(s1, s2, solver, start_ids, x_vars, y_vars, State, Test, Out, inputs_number)\n")
	mainFile.write("\t\t\tsolver_check = solver.check()\n")
	mainFile.write("\t\t\tif solver_check != z3.sat:\n\t\t\t\tflag = 0\n\t\t\t\tbreak\n")
	mainFile.write("if (flag == 1):\n\tprint(\"sat\")\nelse:\n\tprint(\"unsat\")")        
	            #     solver_model = solver.model()
	            #     print "start_ids[", s1, "][", s2, "] = ", solver_model[start_ids[s1][s2]]
	            #     print "start_ids[", s2, "][", s1, "] = ", solver_model[start_ids[s2][s1]]
	            #     print "x_vars[s1][s2] = ", solver_model[x_vars[s1][s2]]
	            # else:
	            #     "unsat for s1 = ", s1, " and s2 = ", s2
	            # print "_______________________"

	mainFile.close()


#не доделан финальный вариант
''' 
'''
def makeFormulaForH(mainFileName, fsm, inputs, outputs, states, SCSet):
	# SCSet = fsm.getSCSet(inputs)
	inputs = [fsm.numberOfInputs if x==-1 else x for x in inputs]
	states = [fsm.numberOfStates if x==-1 else x for x in states]
	outputs = [fsm.numberOfOutputs if x==-1 else x for x in outputs]	
	mainFile = open(mainFileName, 'w')
	mainFile.write ("#!/usr/bin/python\nfrom z3 import *\n\n")
	mainFile.write ("class SolverForTests:\n")
	mainFile.write("\tdef __init__(self, states_number, inputs_number):\n\t\tself.states_number = states_number\n\t\tself.inputs_number = inputs_number\n\t\tself.solver = Solver()\n\t\tself.solver.set(\"timeout\", 7200000)\n\t\tself.StateCover = dict()\n")
	mainFile.write ("\t\tself.Test_seq = [")
	for i in range (len(inputs) - 1):
		mainFile.write (str(inputs[i]) + ", ")
	mainFile.write (str(inputs[-1]) + "]\n")
	mainFile.write ("\t\tself.State_seq = [")
	for i in range (len(states) - 1):
		mainFile.write (str(states[i]) + ", ")
	mainFile.write (str(states[-1]) + "]\n")
	mainFile.write ("\t\tself.Out_seq = [")
	for i in range (len(outputs) - 1):
		mainFile.write (str(outputs[i]) + ", ")
	mainFile.write (str(outputs[-1]) + "]\n")
	mainFile.write("\t\tself.Test = Array('Test', IntSort(), IntSort())\n\t\tself.State = Array('State', IntSort(), IntSort())\n\t\tself.Out = Array('Out', IntSort(), IntSort())\n\t\tfor i in range(0, len(self.Test_seq)):\n\t\t\tself.Test = Store(self.Test, i, self.Test_seq[i])\n\t\tfor i in range(0, len(self.Test_seq)):\n\t\t\tself.State = Store(self.State, i, self.State_seq[i])\n\t\tfor i in range(0, len(self.Test_seq)):\n\t\t\tself.Out = Store(self.Out, i, self.Out_seq[i])\n\t\tself.newStatesDict = dict()\n\t\treturn\n\n")

	# for q in SCSet:
	# 	print (q)


	mainFile.write("\tdef fill_StateCover(self):\n\t\tself.StateCover_len = [")
	for i in SCSet[:-1]:
		totallength = 0
		for j in i:
			totallength += len(i[j])
		mainFile.write(str(totallength) + ', ')
	totallength = 0
	for j in SCSet[-1]:
		totallength = totallength + len(SCSet[-1][j])
	mainFile.write(str(totallength) + "]\n")


	# mainFile.write("\tdef fill_StateCover(self):\n\t\tself.StateCover_len = [")
	# for w in range(fsm.numberOfStates - 1):
	# 	mainFile.write('1, ')
	# mainFile.write('1]\n')




	mainFile.write("\t\tfor i in range(0, " + str(fsm.numberOfStates) + "):\n\t\t\tself.StateCover[i] = Array('SC_%d' % (i), IntSort(), ArraySort(IntSort(), IntSort()))\n")
	for state in range (len(SCSet)):
		seqnumber = 0
		for j in SCSet[state]:	#j это ключ словаря
			for seq in SCSet[state][j]: #идем по словарю
				k = 0
				mainFile.write("\t\tself.solver.add(self.StateCover[%d][%d][%d] == %d)\n" % (state, seqnumber, k, j))
				k+=1
				for val in seq:
					mainFile.write("\t\tself.solver.add(self.StateCover[%d][%d][%d] == %d)\n" % (state, seqnumber, k, int(val)))
					k+=1
				seqnumber+=1
		mainFile.write("\t\t#########\n")
	mainFile.write("\t\treturn\n\n")




	mainFile.write("\tdef fill_newStatesDict(self):\n")
	for state in range(fsm.numberOfStates):
		nxtstateslist = []
		for input_ in range (fsm.numberOfInputs):
			for transition in fsm.transitionList:
				if (str(transition[0]) == str(state) and str(transition[1]) == str(input_)):
					nxtstateslist.append(transition[2])
					break
			else:
				print ("Error! FSM seems to be bad")
		mainFile.write("\t\tself.newStatesDict[%d] = [" % state)
		for ns in nxtstateslist[:-1]:
			mainFile.write("%s, " % ns)
		mainFile.write("%s]\n" % nxtstateslist[-1])
	mainFile.write("\t\treturn\n\n")

	mainFile.write("\tdef create_env(self):\n\t\tlen_test = len(self.Test_seq)\n\t\tself.J_index = [Int('J_index_%d' % (i)) for i in range(0, self.states_number)]\n\t\tself.x_vars = [[Int('x_vars_%d_%d' % (i, j)) for j in range(0, self.states_number) ] \n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.q_vars = [[Int('q_vars_%d_%d' % (i, j)) for j in range(0, self.states_number) ] \n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.st_vars = [[Int('st_vars_%d_%d' % (i, j)) for j in range(0, self.states_number) ] \n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.seqs = list()\n\t\tself.lengths = list()\n\t\tfor i in range(0, self.states_number):\n\t\t\tself.solver.add(self.J_index[i] >= 0, self.J_index[i] < self.StateCover_len[i])\n\t\t\tself.seqs.append(self.StateCover[i][self.J_index[i]])\n\t\t\tself.lengths.append(self.seqs[i][0])\n\n")
	mainFile.write("\t\tfor i in range(0, self.states_number):\n\t\t\tfor j in range(i + 1, self.states_number):\n\t\t\t\tx_i_j = self.x_vars[i][j]\n\t\t\t\tx_j_i = self.x_vars[j][i]\n\t\t\t\tst_i_j = self.st_vars[i][j]\n\t\t\t\tst_j_i = self.st_vars[j][i]\n\t\t\t\tq_i_j = self.q_vars[i][j]\n\t\t\t\tself.solver.add(q_i_j >= 0, q_i_j < self.states_number - 1)\n\t\t\t\tself.solver.add(x_i_j >= 0, x_i_j + self.lengths[i] + q_i_j < len_test)\n\t\t\t\tself.solver.add(x_j_i >= 0, x_j_i + self.lengths[j] + q_i_j < len_test)\n\t\t\t\tself.solver.add(st_i_j == x_i_j + self.lengths[i])\n\t\t\t\tself.solver.add(st_j_i == x_j_i + self.lengths[j])\n\t\treturn\n\n")
	mainFile.write("\tdef create_tran_env(self):\n\t\tself.x2_vars_left = [[[Int('x2_vars_left_%d_%d_%d' % (i, j, k)) for k in range(0, self.states_number) ] \n\t\t\t\tfor j in range(0, self.inputs_number) ]\n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.x2_vars_right = [[[Int('x2_vars_right_%d_%d_%d' % (i, j, k)) for k in range(0, self.inputs_number) ] \n\t\t\t\tfor j in range(0, self.states_number) ]\n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.q2_vars = [[[Int('q2_vars_%d_%d_%d' % (i, j, k)) for k in range(0, self.states_number) ] \n\t\t\t\tfor j in range(0, self.inputs_number) ]\n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.st2_vars_left = [[[Int('st2_vars_left_%d_%d_%d' % (i, j, k)) for k in range(0, self.states_number) ] \n\t\t\t\tfor j in range(0, self.inputs_number) ]\n\t\t\t\tfor i in range(0, self.states_number) ]\n\t\tself.st2_vars_right = [[[Int('st2_vars_right_%d_%d_%d' % (i, j, k)) for k in range(0, self.inputs_number) ] \n\t\t\t\tfor j in range(0, self.states_number) ]\n\t\t\t\tfor i in range(0, self.states_number) ]\n\n")
	mainFile.write("\tdef create_formula(self):\n\t\tlen_test = len(self.Test_seq)\n\t\tfor i in range(0, self.states_number):\n\t\t\tfor j in range(i + 1, self.states_number):\n\t\t\t\tq_i_j = self.q_vars[i][j]\n\t\t\t\tz_i_j = Int('z_%%d_%%d' %% (i, j))\n\t\t\t\tz_j_i = Int('z_%%d_%%d' %% (j, i))\n\t\t\t\tx_i_j = self.x_vars[i][j]\n\t\t\t\tx_j_i = self.x_vars[j][i]\n\t\t\t\tst_i_j = self.st_vars[i][j]\n\t\t\t\tst_j_i = self.st_vars[j][i]\n\t\t\t\tself.solver.add(self.State[x_i_j] == 0)\n\t\t\t\tf1 = ForAll([z_i_j],\n\t\t\t\t\t\tImplies(\n\t\t\t\t\t\t\tAnd(z_i_j >= 0, z_i_j < self.lengths[i]),\n\t\t\t\t\t\t\t\tself.Test[x_i_j + z_i_j] == self.seqs[i][z_i_j + 1]\n\t\t\t\t\t))\n\t\t\t\tself.solver.add(self.State[x_j_i] == 0)\n\t\t\t\tf2 = ForAll([z_j_i],\n\t\t\t\t\t\tImplies(\n\t\t\t\t\t\t\tAnd(z_j_i >= 0, z_j_i < self.lengths[j]),\n\t\t\t\t\t\t\t\tself.Test[x_j_i + z_j_i] == self.seqs[j][z_j_i + 1]\n\t\t\t\t\t))\n\t\t\t\tself.solver.add(f1)\n\t\t\t\tself.solver.add(f2)\n\t\t\t\tf3 = And(st_i_j + q_i_j < len_test,\n\t\t\t\t\t\t self.Test[st_i_j + q_i_j] != %d,\n\t\t\t\t\t\t self.Test[st_j_i + q_i_j] != %d,\n\t\t\t\t\t\t self.Test[st_i_j + q_i_j] == self.Test[st_j_i + q_i_j],\n\t\t\t\t\t\t self.Out[st_i_j + q_i_j] != self.Out[st_j_i + q_i_j])\n\t\t\t\tself.solver.add(f3)\n\t\t\t\tt_i_j = Int('t_%%d_%%d' %% (i, j))\n\t\t\t\tt_constr = And(t_i_j >= 0, t_i_j < q_i_j,\n\t\t\t\t\t\t\t   self.Test[st_i_j + t_i_j] != %d,\n\t\t\t\t\t\t\t   self.Test[st_j_i + t_i_j] != %d,\n\t\t\t\t\t\t\t   self.Test[st_i_j + t_i_j] == self.Test[st_j_i + t_i_j],\n\t\t\t\t\t\t\t   self.Out[st_i_j + t_i_j] == self.Out[st_j_i + t_i_j])\n\t\t\t\tf4 = ForAll([t_i_j],\n\t\t\t\t\t\tImplies(And(t_i_j >= 0, t_i_j < q_i_j),\n\t\t\t\t\t\t\t\tAnd(self.Test[st_i_j + t_i_j] != %d,\n\t\t\t\t\t\t\t\t\tself.Test[st_j_i + t_i_j] != %d,\n\t\t\t\t\t\t\t\t\tself.Test[st_i_j + t_i_j] == self.Test[st_j_i + t_i_j],\n\t\t\t\t\t\t\t\t\tself.Out[st_i_j + t_i_j] == self.Out[st_j_i + t_i_j])))\n\t\t\t\tself.solver.add(f4)\n\t\treturn\n\n" % (fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs))
	mainFile.write('''\tdef industrial_transition_formula(self):\n\t\tlen_test = len(self.Test_seq)\n\t\ts = 0\n\t\ti = 0\n\t\tfor s in range(0, self.states_number):\n\t\t\tfor i in range(0, self.inputs_number):\n\t\t\t\tfor k in range(0, self.states_number):\n\t\t\t\t\tnew_state = self.newStatesDict[s][i]\n\t\t\t\t\tif (k != new_state):\n\t\t\t\t\t\tx_s_i_k = self.x2_vars_left[s][i][k]\n\t\t\t\t\t\tx_k_s_i = self.x2_vars_right[k][s][i]\n\t\t\t\t\t\tst_s_i_k = self.st2_vars_left[s][i][k]\n\t\t\t\t\t\tst_k_s_i = self.st2_vars_right[k][s][i]\n\t\t\t\t\t\tq = self.q2_vars[s][i][k]\n\t\t\t\t\t\tz_s = Int('z_left_%%d_%%d_%%d' %% (s, i, k))\n\t\t\t\t\t\tz_k = Int('z_right_%%d_%%d_%%d' %% (k, s, i))\n\t\t\t\t\t\tt = Int('t_%%d_%%d_%%d' %% (s, i, k))\n\t\t\t\t\t\t##################################\n\t\t\t\t\t\tself.solver.add(q >= 0, q < self.states_number - 1)\n\t\t\t\t\t\tself.solver.add(x_s_i_k >= 0, x_s_i_k + self.lengths[s] + 1 + q < len_test)\n\t\t\t\t\t\tself.solver.add(self.State[x_s_i_k] == 0)\n\t\t\t\t\t\tf1 = ForAll([z_s],\n\t\t\t\t\t\t\tImplies(\n\t\t\t\t\t\t\t\tAnd(z_s >= 0, z_s < self.lengths[s]),\n\t\t\t\t\t\t\t\t\tself.Test[x_s_i_k + z_s] == self.seqs[s][z_s + 1]\n\t\t\t\t\t\t))\n\t\t\t\t\t\tself.solver.add(x_k_s_i >= 0, x_k_s_i + self.lengths[k] + q < len_test)\n\t\t\t\t\t\tself.solver.add(self.State[x_k_s_i] == 0)\n\t\t\t\t\t\tf2 = ForAll([z_k],\n\t\t\t\t\t\t\tImplies(\n\t\t\t\t\t\t\t\tAnd(z_k >= 0, z_k < self.lengths[k]),\n\t\t\t\t\t\t\t\t\tself.Test[x_k_s_i + z_k] == self.seqs[k][z_k + 1]\n\t\t\t\t\t\t))\n\t\t\t\t\t\tself.solver.add(f1)\n\t\t\t\t\t\tself.solver.add(f2)\n\t\t\t\t\t\t\n\t\t\t\t\t\tself.solver.add(st_s_i_k == x_s_i_k + self.lengths[s] + 1)\n\t\t\t\t\t\t\n\t\t\t\t\t\tself.solver.add(st_k_s_i == x_k_s_i + self.lengths[k])\n\t\t\t\t\t\t\n\t\t\t\t\t\tself.solver.add(self.Test[x_s_i_k + self.lengths[s]] == i)\n\t\t\t\t\t\t\n\t\t\t\t\t\tf3 = And(st_s_i_k + q < len_test,\n\t\t\t\t\t\t\t\t self.Test[st_s_i_k + q] != %d,\n\t\t\t\t\t\t\t\t self.Test[st_k_s_i + q] != %d,\n\t\t\t\t\t\t\t\t self.Test[st_s_i_k + q] == self.Test[st_k_s_i + q],\n\t\t\t\t\t\t\t\t self.Out[st_s_i_k + q] != self.Out[st_k_s_i + q])\n\t\t\t\t\t\tself.solver.add(f3)\n\t\t\t\t\t\tf4 = ForAll([t],\n\t\t\t\t\t\t\t\tImplies(And(t >= 0, t < q),\n\t\t\t\t\t\t\t\t\t\tAnd(self.Test[st_s_i_k + t] != %d,\n\t\t\t\t\t\t\t\t\t\t\tself.Test[st_k_s_i + t] != %d,\n\t\t\t\t\t\t\t\t\t\t\tself.Test[st_s_i_k + t] == self.Test[st_k_s_i + t],\n\t\t\t\t\t\t\t\t\t\t\tself.Out[st_s_i_k + t] == self.Out[st_k_s_i + t])))\n\t\t\t\t\t\tself.solver.add(f4)\n\t\treturn\n\n''' % (fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs))
	mainFile.write("\tdef create_transition_formula(self):\n\t\tlen_test = len(self.Test_seq)\n\t\t#for k in range(1, self.states_number):\n\t\tk = 1\n\t\ti = 0\n\t\tx_0_0_k = self.x2_vars[0][0][k]\n\t\tx_k_0_0 = self.x2_vars[k][0][0]\n\t\tst_0_0_k = self.st2_vars[0][0][k]\n\t\tst_k_0_0 = self.st2_vars[k][0][0]\n\t\tq = self.q2_vars[0][0][k]\n\t\tz_0 = Int('z_0_0_%%d' %% (k))\n\t\tz_k = Int('z_%%d_0_0' %% (k))\n\t\tt = Int('t_0_0_%%d' %% (k))\n\t\t##################################\n\t\tself.solver.add(q >= 0, q < self.states_number - 1)\n\t\tself.solver.add(x_0_0_k >= 0, x_0_0_k + self.lengths[0] < len_test)\n\t\tself.solver.add(self.State[x_0_0_k] == 0)\n\t\tf1 = ForAll([z_0],\n\t\t\tImplies(\n\t\t\t\tAnd(z_0 >= 0, z_0 < self.lengths[0]),\n\t\t\t\t\tself.Test[x_0_0_k + z_0] == self.seqs[0][z_0 + 1]\n\t\t))\n\t\tself.solver.add(x_k_0_0 >= 0, x_k_0_0 + self.lengths[k] < len_test)\n\t\tself.solver.add(self.State[x_k_0_0] == 0)\n\t\tf2 = ForAll([z_k],\n\t\t\tImplies(\n\t\t\t\tAnd(z_k >= 0, z_k < self.lengths[k]),\n\t\t\t\t\tself.Test[x_k_0_0 + z_k] == self.seqs[k][z_k + 1]\n\t\t))\n\t\tself.solver.add(f1)\n\t\tself.solver.add(f2)\n\t\t#self.solver.add(self.Test[x_0_0_k + self.lengths[0] + 1] == i)\n\t\t\n\t\tself.solver.add(st_0_0_k == x_0_0_k + self.lengths[0] + 1)\n\t\tself.solver.add(st_k_0_0 == x_k_0_0 + self.lengths[k])\n\t\tself.solver.add(self.Test[st_0_0_k] == i)\n\n\t\tf3 = And(st_0_0_k + q < len_test,\n\t\t\t\t self.Test[st_0_0_k + q] != %d,\n\t\t\t\t self.Test[st_k_0_0 + q] != %d,\n\t\t\t\t self.Test[st_0_0_k + q] == self.Test[st_k_0_0 + q],\n\t\t\t\t self.Out[st_0_0_k + q] != self.Out[st_k_0_0 + q])\n\t\tself.solver.add(f3)\n\t\t\n\t\tf4 = ForAll([t],\n\t\t\t\tImplies(And(t >= 0, t < q),\n\t\t\t\t\t\tAnd(self.Test[st_0_0_k + t] != %d,\n\t\t\t\t\t\t\tself.Test[st_k_0_0 + t] != %d,\n\t\t\t\t\t\t\tself.Test[st_0_0_k + t] == self.Test[st_k_0_0 + t],\n\t\t\t\t\t\t\tself.Out[st_0_0_k + t] == self.Out[st_k_0_0 + t])))\n\t\tself.solver.add(f4)\n\t\t\n\t\treturn\n\n\n" % (fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs, fsm.numberOfInputs))

	# mainFile.write('''\tdef check_formula(self):\n\t\tself.check = self.solver.check()\n\t\tprint(self.check)\n\t\tif (self.check == z3.sat):\n\t\t\tself.model = self.solver.model()\n\t\t\tfor i in range(0, self.states_number):\n\t\t\t\tprint("j%d = " % (i), self.model[self.J_index[i]])\n\t\t\tfor i in range(0, self.states_number):\n\t\t\t\tfor j in range(i + 1, self.states_number):\n\t\t\t\t\tprint("________________________________")\n\t\t\t\t\tprint("q_%d_%d = " % (i, j), self.model[self.q_vars[i][j]])\n\t\t\t\t\tprint("x_%d_%d = " % (i, j), self.model[self.x_vars[i][j]])\n\t\t\t\t\tprint("x_%d_%d = " % (j, i), self.model[self.x_vars[j][i]])\n\t\t\t\t\tprint("st_%d_%d = " % (i, j), self.model[self.st_vars[i][j]])\n\t\t\t\t\tprint("st_%d_%d = " % (j, i), self.model[self.st_vars[j][i]])\n\t\t\tfor k in range(0, self.states_number):\n\t\t\t\tprint("####################################")\n\t\t\t\tprint("q2_0_0_%d = " % (k), self.model[self.q2_vars[0][0][k]])\n\t\t\t\tprint("x2_l_0_0_%d = " % (k), self.model[self.x2_vars_left[0][0][k]])\n\t\t\t\tprint("x2_r_%d_0_0 = " % (k), self.model[self.x2_vars_right[k][0][0]])\n\t\t\t\tprint("st2_l_0_0_%d = " % (k), self.model[self.st2_vars_left[0][0][k]])\n\t\t\t\tprint("st2_r_%d_0_0 = " % (k), self.model[self.st2_vars_right[k][0][0]])\n\t\treturn\n''')
	# mainFile.write('''\tdef verify_z3(self):\n\t\tverif = True\n\t\tfor i in range(0, self.states_number):\n\t\t\tfor j in range(i + 1, self.states_number):\n\t\t\t\tJ_i = int(str(self.model[self.J_index[i]]))\n\t\t\t\tJ_j = int(str(self.model[self.J_index[j]]))\n\t\t\t\tx_i_j = int(str(self.model[self.x_vars[i][j]]))\n\t\t\t\tx_j_i = int(str(self.model[self.x_vars[j][i]]))\n\t\t\t\tq_i_j = int(str(self.model[self.q_vars[i][j]]))\n\t\t\t\tst_i_j = int(str(self.model[self.st_vars[i][j]]))\n\t\t\t\tst_j_i = int(str(self.model[self.st_vars[j][i]]))\n\t\t\t\tif ((self.State_seq[x_i_j] == 0) and (self.State_seq[x_j_i] == 0)):\n\t\t\t\t\tif ((self.State_seq[st_i_j] == i) and (self.State_seq[st_j_i] == j)):\n\t\t\t\t\t\tfor k in range(0, q_i_j):\n\t\t\t\t\t\t\tif ((self.Test_seq[st_i_j + k] != self.Test_seq[st_j_i + k]) or\n\t\t\t\t\t\t\t\t(self.Out_seq[st_i_j + k] != self.Out_seq[st_j_i + k])\n\t\t\t\t\t\t\t\t):\n\t\t\t\t\t\t\t\tprint("error")\n\t\t\t\t\tif (self.Out_seq[st_i_j + q_i_j] == self.Out_seq[st_j_i + q_i_j]):\n\t\t\t\t\t\tprint("error")\n\t\t\t\telse:\n\t\t\t\t\tverif = False\n\t\tprint("verif = ", verif)\n\t\treturn\n''')

	mainFile.write ("test = SolverForTests(%d, %d)\n" % (fsm.numberOfStates, fsm.numberOfInputs))
	mainFile.write("test.fill_StateCover()\ntest.fill_newStatesDict()\ntest.create_env()\ntest.create_tran_env()\n\ntest.create_formula()\n#test.new_create_transition_formula()\ntest.industrial_transition_formula()\n\n")
	mainFile.write ('if (test.solver.check() == z3.sat):\n\tprint("sat")\nelse:\n\tprint ("unsat")\n')
	mainFile.close()


if __name__ == '__main__':
	parser = createParamParser()
	args = parser.parse_args()
	mainFileName = args.mainFile
	fsm = FSM ()
	fsm.readFSMfromFile(args.fsmFile)
	# fsm.printTransitionTable()
	(inputs, outputs) = readTestFromFile(args.testFile)
	states = []
	(states, outputs) = fsm.getStatesAndOutputsFromTest(inputs)

	# print ("inp")
	# print (inputs)
	# print ("outp")
	# print (outputs)
	# print ("states")
	# print (states)
	# for i in range(len(inputs)):
	# 	print (str(states[i]) + " " + str(inputs[i]) + "/" + str(outputs[i]))
	

	if args.test_type == "TT":
		makeFormulaForTT(mainFileName, fsm, inputs, states)
		# fsm.checkTransitionsCovering(args.testFile)
	elif args.test_type == "W":
		makeFormulaForW(mainFileName, fsm, inputs, outputs, states)
	elif args.test_type == "H":
		makeFormulaForH(mainFileName, fsm, inputs, outputs, states)
	else:
		print ("error test type")
