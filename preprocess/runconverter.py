#gaudipython input used to convert dst file to json

from Gaudi.Configuration import *

from Configurables import DaVinci

from Configurables import FilterDesktop

from StandardParticles import StdAllNoPIDsPions

from Configurables import JsonConverter

myconverter = JsonConverter()

DaVinci().EventPreFilters = [ ]

DaVinci().UserAlgorithms = [ myconverter ]

wdir_apanin = "/afs/cern.ch/work/a/apanin/"
wdir_derkach = "/afs/cern.ch/work/d/derkach/"
import os

DaVinci().Input = [os.path.join( wdir_derkach,"public/00036954_00000005_1.allstreams.dst")]


DaVinci().DataType = "2012"
DaVinci().Simulation = False

DaVinci().PrintFreq = 1

#DaVinci().DDDBtag = "dddb-20130111"
#DaVinci().CondDBtag = "cond-20130111"
DaVinci().CondDBtag = "cond-20140604"
DaVinci().DDDBtag = "dddb-20130929-1"
