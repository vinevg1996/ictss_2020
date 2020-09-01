# ictss_2020

The main script for running experiments is a "exp.py" (generating FSMs, constructing formulas and checking them)
FSM.py is a class for manipulating FSMs
makeFormula.py is a script for making z3 formula based on a FSM and a test suite

exp.py
	set values for theese variables
		workingDir - path to a directory where all files (python scripts, generated FSMs, etc...) for experiments are stored
		pathToFSMGenerator - path to a binary FSM generator
		pathToTestGenerator - transitionTour or a test_fsm.exe (running using wine)

	to carrie out an experiment with formulas for checking transitions covering set following values of corresponding variables
		testType = "TT"
		testexp = "complete" # for checking complete test suites
		testexp = "notcomplete" # for checking not complete test suites
		FSMs, test suites and corresponding formulas will be generated in directories  <workingDir>/FSMsTTcomplete, <workingDir>/FSMsTTnotcomplete, make shure these directories exist before running exp.py

	to carrie out an experiment with formulas for checking formulas according Proposition 2 of the paper, set following values of corresponding variables
		testType = "H"
		testexp = "complete" or "notcomplete" as in previous type of experiment
		FSMs, test suites and corresponding formulas will be generated in directories  <workingDir>/FSMsHcomplete, <workingDir>/FSMsHnotcomplete, make shure these directories exist before running exp.py


	number of generated FSMs depends on loops in lines 148 and 149 and the value of a variable "fsmsNum"
	also make sure directorie <workingdir>/logs exist for generating log files with results of experiments
