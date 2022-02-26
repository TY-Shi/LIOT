import numpy as np
from mogutda import SimplicialComplex

e1 = [[255,255], [255,0]]
sc1 = SimplicialComplex(e1)
print (sc1.betti_number(0))
