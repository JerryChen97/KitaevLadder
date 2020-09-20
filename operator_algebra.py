import numpy as np
from itertools import product

def comm(i1:int, i2:int):
    available_set = [0, 1, 2, 3]
    assert i1 in available_set
    assert i2 in available_set
    if i1 * i2 == 0:
        return 1
    elif i1 == i2:
        return 1
    else:
        return -1

class Op:
    def __init__(self, ops:list, shift:int = 0):
        for op in ops:
            assert op==0 or op==1 or op==2 or op==3
        self.ops = ops
        assert type(shift) is int
        assert shift >= 0
        
        # standardization
        if ops == []:
            shift = 0
            pass
        else:
            while ops[0] == 0:
                shift += 1
                ops.pop(0)
                pass
            while ops[-1] == 0:
                ops.pop(-1)
                pass
        
        self.shift = shift
        self.start_site = self.shift
        self.end_site = self.start_site + len(ops)
        pass 
    
    def __str__(self):
        if self.shift==0:
            return str(self.ops)
        else:
            return str([0 for i in range(self.shift)] + self.ops)
        pass
    
    def init_from_str(self, opstr:str):
        self.__init__([int(ch) for ch in str])
        pass
    
    def get_full_ops(self):
        return [0 for i in range(self.shift)] + self.ops
    
    def to_str(self):
        if self.shift==0:
            return ''.join([str(i) for i in self.ops])
        else:
            return ''.join('0' for i in range(self.shift)) + ''.join([str(j) for j in self.ops])
        pass
    
    """
        Convert the string list to a standard version: (op_str, position)
    """
    # def to_std(self):
        
    """
        Function return whether the Op commute with another (op2):
            return 1 when commutative
            return -1 when anti-commutative
    """
    def commute_with(self, op2) -> int:
        
        # first of all, we only consider the region that the two op_list overlap with each other
        if self.start_site >= op2.end_site or op2.start_site >= self.end_site:
            return 1
        else:
            start_site = max(self.start_site, op2.start_site)
            end_site = min(self.end_site, op2.end_site)
            comm_list = [0 for i in range(start_site, end_site)]
#             print(comm_list)
            for i in range(start_site, end_site):
                site1 = i - self.shift
                site2 = i - op2.shift
                comm_list[i-start_site] = comm(self.ops[site1],op2.ops[site2])
                pass
            return np.prod(comm_list)
        pass
    
    """ 
        Test whether or not this operator commute with a list of other ops.
        Main purpose is to check if the local symmetries are satisfied
    """
    def commute_with_list(self, op_list) -> bool:
        for op in op_list:
            if self.commute_with(op) == -1:
                return False
        return True
    
    """ 
        Test whether of note this operator anticomm with a list of other ops.
        Main purpose is to check if the local order parameter condition (anti-commutation with Hamiltonians) is satisfied
    """
    def anti_commute_with_list(self, op_list) -> bool:
        for op in op_list:
            if self.commute_with(op) == 1:
                return False
        return True

""" 
    Generate the local symmetries we used in Kitaev ladders
"""
def gen_local_sym_D(domin='x', shift=0):
    assert domin=='x' or domin=='y'
    if domin=='x':
        return Op([1, 2, 2, 1], shift=shift)
    else:
        return Op([2, 1, 1, 2], shift=shift)
    
""" 
    Generate the local symmetries we used in Kitaev ladders in a specific range.
    `srange` to specify the unitcell range to consider the local symmetries `Dn`
"""
def gen_local_sym_D_range(domin='x', srange=4, bc='open'):
    assert domin=='x' or domin=='y'
    assert bc=='open' # currently we don't consider the periodic bc
    if domin=='x':
        bulk_symmetries = [Op([1, 2, 2, 1], shift=2*shift) for shift in range(srange)]
        if bc=='open':
            edge_symmetries = [Op([2, 1]), Op([1, 2], shift=2*srange)] 
    else:
        bulk_symmetries = [Op([2, 1, 1, 2], shift=2*shift) for shift in range(srange)]
        if bc=='open':
            edge_symmetries = [Op([1, 2]), Op([2, 1], shift=2*srange)] 
    return bulk_symmetries + edge_symmetries
"""
    Generate the Hamiltonian we used in Kitaev ladders.
    The model we will consider is defined on the 1D chain.
"""
def gen_Hamiltonian(domin='x', bc='open', max_N_unitcell=8):
    assert domin=='x' or domin=='y'
    
    first_bond = 3
    second_bond = 1 if domin=='x' else 2
    third_bond = 2 if domin=='x' else 1
    
    first_bond_set = [Op(ops=[first_bond, first_bond], shift=2*k) for k in range(max_N_unitcell)]
    second_bond_set = [Op(ops=[second_bond, second_bond], shift=1+2*k) for k in range(max_N_unitcell)]
    third_bond_set = [Op(ops=[third_bond, 0, 0, third_bond], shift=2*k) for k in range(max_N_unitcell)]
    
    return first_bond_set + second_bond_set + third_bond_set
    
"""
    Generate all the possible local operators to some extent that we specify
"""
def gen_local_ops(
    local_range=1, # the number of unicells involved
):
    assert type(local_range) is int
    assert local_range >= 1
    all_ops = []
    if local_range == 1: # all the possible local ops occupying only one unitcell
        non_trivial = [1, 2, 3]
        for op in non_trivial:
            all_ops.append(Op([op, 0]))
        for op in non_trivial:
            all_ops.append(Op([0, op]))
        for op1 in non_trivial:
            for op2 in non_trivial:
                all_ops.append(Op([op1, op2]))
    else:
        one_cell_ops = gen_local_ops(local_range=1) # the base ops
        possible_op = [0, 1, 2, 3] # the candidates for the descending op
        iter_res = list(product(possible_op, repeat=2*(local_range - 1)))
        
        
        for op in one_cell_ops:
            prefix = op.get_full_ops()
            if len(prefix)==1: prefix += [0]
#             print(f"prefix: {prefix}")
            for t in iter_res:
                suffix = list(t)
#                 print(f"    suffix: {suffix}")
                op_to_append = Op(prefix+suffix)
#                 print(f"    op: {op_to_append}")
                all_ops.append(op_to_append)
                pass
            pass
    return all_ops

def gen_global_sym_S_range(srange=4):
    X = Op(ops=[1 for k in range(srange*2 + 2)])
    Y = Op(ops=[2 for k in range(srange*2 + 2)])
    Z1= Op(ops = [3 if (k % 4 == 1 or k % 4 == 2) else 0 for k in range(srange*2 + 2)])
    Z2= Op(ops = [0 if (k % 4 == 1 or k % 4 == 2) else 3 for k in range(srange*2 + 2)])
    
    return [X, Y, Z1, Z2]

def gen_local_order_parameter_candidates(range_int=4):
    D_list = gen_local_sym_D_range(srange=range_int)
    S_list = gen_global_sym_S_range(srange=range_int)
    
    candidate_list = []
    for op in gen_local_ops(local_range=range_int):
        if op.commute_with_list(D_list):
            if op.commute_with(S_list[0])==-1 or op.commute_with(S_list[1])==-1:
                candidate_list.append(op)
    return candidate_list
