import itertools
from itertools import combinations

import numpy as np

def odd_comb(input_list):
    return [list(combinations(input_list, n))]