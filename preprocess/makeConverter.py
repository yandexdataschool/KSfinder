#i generate prequesites for multi-dst json conversion

wdir_apanin = "/afs/cern.ch/work/a/apanin/"
wdir_derkach = "/afs/cern.ch/work/d/derkach/"
import os
datadir = wdir_apanin+"bg_dst/down"
nunus = os.listdir(datadir)



src = """
from Gaudi.Configuration import *

from Configurables import DaVinci

from Configurables import FilterDesktop

from StandardParticles import StdAllNoPIDsPions

from Configurables import JsonConverter

myconverter = JsonConverter()

DaVinci().EventPreFilters = [ ]

DaVinci().UserAlgorithms = [ myconverter ]


DaVinci().Input = [ "{}" ]	

DaVinci().DataType = "2012"
DaVinci().Simulation = False

DaVinci().PrintFreq = 1

#DaVinci().DDDBtag = "dddb-20130111"
#DaVinci().CondDBtag = "cond-20130111"
DaVinci().CondDBtag = "cond-20140604"
DaVinci().DDDBtag = "dddb-20130929-1"
"""
for i in range(len(nunus)):
    os.mkdir("bg"+str(i))
    os.chdir("bg"+str(i))
    rconv_script = src.format(os.path.join(datadir,nunus[i]) )

    with open("runconverter.py",'w') as fout:
        fout.write(rconv_script)

    os.chdir("..")

runscript = """
for i in {0..%s}; do 
    cd "bg"$i
    GaudiRun "runconverter.py";
    cd ..
done
"""%len(nunus)
print runscript
