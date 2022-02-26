"""
    Module dedicated to the Persistence Diagram (PD)
"""

import numpy as np

def save_PD(pd,name="pers_diag.txt"):
    """
        A method for saving a persistence diagram as a txt file

        :param pd: the persistence diagram
        :param name:
        :type name: string
    """

    np.savetxt(name,pd)

def load_PD(name="pers_diag.txt"):
    """
        A method for loading a persistence diagram from a txt file

        :param name:
        :type name: string

        :returns: The persistence diagram
    """
    return np.loadtxt(name)

def remove_inf(pd):
    """
        A method for removing all the tuples containing an infinite value from a persistence diagram

        :param pd: The persistence diagram

        :returns: The cleaned persistence diagram
    """
    print (type(pd))
    tmp = list(map(lambda row: ~np.isinf(row).any(), pd))
    return pd[tmp]


def rescale(pd,t_range=[]):
    """
        A method for rescaling a persistence diagram within a specific interval.

        :param pd: The persistence diagram

        :param t_range: pair of values indicating the minimum and maximum filtration in which rescaling the entire diagram
        :type t_range: np.array

        :returns: The rescaled persistence diagram
    """

    if(len(pd.shape) == 1):
        #this is the case where the persistence diagrams has only one point
        pd = np.array([pd])

    t_min=t_max=0

    if len(t_range) == 2:
        t_min = t_range[0]
        t_max = t_range[1]
    else:
        t_min = pd.T[1].min()
        t_max = pd.T[2].max()

    return np.array(map(lambda row: [row[0], (row[1] - t_min)/(t_max - t_min), (row[2] - t_min)/(t_max - t_min)], pd))


def mask(pd, value, t_feature):
    """
        A method for modifying a persistence diagram by filtering out some of the persistence dots.

        :param pd: The persistence diagram to filter

        :param t_feature: Type of feature for filtering the persistence diagram. One among

        * 'birth<' - for filtering all the points with birth lower than value

        * 'birth>' - for filtering all the points with birth higher than value

        * 'death<' - for filtering all the points with death lower than value

        * 'death>' - for filtering all the points with birth higher than value

        * 'dim' - for filtering all the points with dimension different than value

        * 'life' - for filtering all the points with a lifespan smaller than value

        :param value: the value to use for the filtering feature


        :returns: The filtered persistence diagram
    """

    accepted_features = ['birth<' ,'birth>','death<','death>','dim','life']
    if t_feature not in accepted_features:
        print("INVALID COMMAND: List of accepted features, ",accepted_features)
        return pd

    #keep all the persistence intervals of the desired dimension (0,1 or 2 cycle)
    if t_feature == 'dim':
        return pd[np.array(map(lambda row: row[0] == value, pd))]
    #keep all the persistence intervals with birth time lower than value
    elif t_feature == 'birth<':
        return pd[map(lambda row: row[1] < value, pd)]
    #keep all the persistence intervals with birth time greater than value
    elif t_feature == 'birth>':
        return pd[map(lambda row: row[1] > value, pd)]
    #keep all the persistence intervals with death time lower than value
    elif t_feature == 'death<':
        return pd[map(lambda row: row[2] < value, pd)]
    #keep all the persistence intervals with death time greater than value
    elif t_feature == 'death>':
        return pd[map(lambda row: row[2] > value, pd)]
    #keep all the persistence intervals with life span greater than value
    elif t_feature == 'life':
        return pd[map(lambda row: row[2]-row[1] > value, pd)]
    else:
        print("Should not be here")
        return pd
