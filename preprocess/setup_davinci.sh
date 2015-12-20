#!/bin/bash
#to be run on lxplus: installs DaVinci via setupproject
SetupProject DaVinci v36r7p7 --build-env 
cd ~/cmtuser/DaVinci_v36r7p7
svn checkout svn+ssh://svn.cern.ch/reps/lhcb/Online/trunk/Online/WebGLDisplay/JsonConverter 
cmt make
SetupProject DaVinci v36r7p7 --dev
