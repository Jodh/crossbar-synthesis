################# Instructions for Edge-detector.py ##################################

##### Files Required ###########################
	1. edge-detector.py
	2. tt_reduced_plus_5.npy (the ground truth table)

##### Libraries Required ########################
	1. numpy

##### To run ####################################
	1. Have file (1) and (2) in same directory
	2. Use `python edge-detector.py`

##### Output ######################################
	1. delta
	2. Design Matrix

##### Output Description ###########################
	1. delta - an integer represeting the number of mismatches between the ground truth table 
and produced design. Lower number is better.
	2. Design Matrix - D where Dij represents a memristor assingment with the following mapping
0: False
1: True
(2, ...,9): literals corresponding to the input bits of a 8-bit pixel X
(10,...,17): literals corresponding to the input bits of a 8-bit pixel Y
(18,..., 25): negation of literals corresponding to the input bits of a 8-bit pixel X
(26,..., 33): negation of literals corresponding to the input bits of a 8-bit pixel Y

##### Modifying the file #############################
	1. function `generate_random_design(n,l,num_input_bits,probability_distribution)` produces a random nxl crossbar design. Comment out line 224 and use line 223 for a 8x8 crossbar.
	2. function `simulated_annealing(...)` produces the designs and prints the output.
