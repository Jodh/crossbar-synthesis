import csv, random, subprocess
import numpy as np

############# function to compute binary addition ##############################
def add_binary_nums(x,y):
    max_len = max(len(x), len(y))

    x = x.zfill(max_len)
    y = y.zfill(max_len)

    result = ''
    carry = 0

    for i in range(max_len-1, -1, -1):
        r = carry
        r += 1 if x[i] == '1' else 0
        r += 1 if y[i] == '1' else 0
        result = ('1' if r % 2 == 1 else '0') + result
        carry = 0 if r < 2 else 1

    # if carry !=0 : result = '1' + result

    return result.zfill(max_len), carry
################################################################################

########## function to generate n-bit bitstrings ###############################
def get_input(nbits):
    out_str = ''
    for i in range(nbits):
        if random.random()> 0.5:
            out_str += '1'
        else:
            out_str += '0'
    return out_str
################################################################################

########## function to convert binary values (0/1) to memr state values (0.00001/1.0)
def bin2state(in_bin):
    if in_bin == '1':
        return '1.0'
    else:
        return '0.00001'
################################################################################

########## function returns negation of '1' or '0' #############################
def get_neg(in_val):
    if in_val == '1':
        return '0'
    else:
        return '1'
################################################################################

########## extract output voltages from output files ###########################
def extract_volts(outrow):
    op_fname = 'opfile.txt'
    search_str = 'm'+str(outrow)+'       '
    with open(op_fname, 'r') as f:
        content = f.readlines()
    for l in content:
        if search_str in l:
            lsize = len(l)
            out_volt = l[int(lsize/2):].strip()
            return out_volt


########## function to map values into crossbar ################################
def xbar_eval(in_1, in_2, in_xbar):
    nbits = len(in_1)
    in_1 = in_1[::-1]
    in_2 = in_2[::-1]
    input_dict = {}
    input_dict[0] = '0.00001'
    input_dict[1] = '1.0'
    n = 2
    bit_count = 0
    nrows, ncols = in_xbar.shape

    while bit_count < nbits:
        input_dict[n] = bin2state(in_1[bit_count])
        n += 1
        input_dict[n] = bin2state(get_neg(in_1[bit_count]))
        n += 1
        input_dict[n] = bin2state(in_2[bit_count])
        n += 1
        input_dict[n] = bin2state(get_neg(in_2[bit_count]))
        n += 1
        bit_count += 1

    xbar_str = ''
    # print(in_xbar, input_dict)
    for i in range(nrows):
        for j in range(ncols):
            tmp_memr = xbar[i][j]
            tmpstr = 'X_'+str(i)+'_'+str(j)+' m'+str(i)+' n'+str(j)+' mem_dev xo='+input_dict[tmp_memr]+'\n'
            xbar_str += tmpstr
    # print(xbar_str)
    with open('subckt_memr.txt', 'r') as f:
        subckt_str = f.read()
    spice_str = '$ adder for '+ str(nbits) +'-bit inputs\n'
    spice_str += subckt_str + '\n\n'+xbar_str+'\n'
    spice_str += 'V1 m0 0 1\n'
    spice_str += 'Rout m'+str(nrows-1)+' 0 100 \n\n\n'
    spice_str += '.control\ntran 1 2\n.endc\n.end'

    # fname_spice = 'carry-'+str(nbits)+'.cir'
    fname_spice = 'spicefileChk.cir'
    with open(fname_spice, 'w') as f:
        f.write(spice_str)
    fname_spiceout = 'opfile.txt'
    bash_cmd = 'ngspice -b '+fname_spice+' -o '+fname_spiceout
    # print(bash_cmd)
    subprocess.call(bash_cmd, shell=True)
    out_volt = extract_volts(nrows-1)
    return out_volt
################################################################################

for bitnum in range(2, 129):
    xbar_str = []
    with open('csv-files-carry/carry-'+str(bitnum)+'.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            xbar_str.append(row)
    nrows = len(xbar_str)
    ncols = len(xbar_str[0])
    print(nrows, ncols)
    xbar = np.zeros((nrows, ncols), dtype=int)
    for i in range(nrows):
        for j in range(ncols):
            xbar[i][j] = int(xbar_str[i][j])

    maxval = 2**bitnum-1
    n_samples = 10
    fname_op = 'outputs/carry-'+str(bitnum)+'.txt'
    for sample_num in range(n_samples):
        a = get_input(bitnum)
        b = get_input(bitnum)
        s, c = add_binary_nums(a, b)
        out_volt = xbar_eval(a, b, xbar)
        with open(fname_op, 'a') as f:
            f.write(a+'\t'+b+'\t'+str(c)+'\t'+out_volt+'\n')
