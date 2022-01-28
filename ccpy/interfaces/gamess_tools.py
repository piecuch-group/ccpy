import numpy as np
from cclib.io import ccread

def parseGamessLogFile(gamessFile, nfrozen):
    """Builds the System object using the SCF information contained within a
       GAMESS log file.

        Arguments:
        ----------
        gamessFile : str -> Path to GAMESS log file
        nfrozen : int -> number of frozen electrons
        Returns:
        ----------
        sys : Object -> System object"""
    from ccpy.models.system import System
    data = ccread(gamessFile)
    return System(data.nelectrons,
               data.nmo,
               data.mult,
               nfrozen,
               getGamessPointGroup(gamessFile),
               data.mosyms[0],
               data.charge)

def getGamessPointGroup(gamessFile):
    """Dumb way of getting the point group from GAMESS log files.

    Arguments:
    ----------
    gamessFile : str -> Path to GAMESS log file
    Returns:
    ----------
    point_group : str -> Molecular point group"""
    point_group = 'C1'
    flag_found = False
    with open(gamessFile, 'r') as f:
        for line in f.readlines():
            if flag_found:
                order = line.split()[-1]
                if len(point_group) == 3:
                    point_group = point_group[0] + order + point_group[2]
                if len(point_group) == 2:
                    point_group = point_group[0] + order
                if len(point_group) == 1:
                    point_group = point_group[0] + order
                break
            if 'THE POINT GROUP OF THE MOLECULE IS' in line:
                point_group = line.split()[-1]
                flag_found = True
    return point_group
