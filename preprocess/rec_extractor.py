#!/usr/bin/env python
from Bender.Main import *
# =============================================================================
#whatdoido: i process damn monte-carlo to get KS->something decays
import traceback
column_names = ['runID','eventID', 'originX','originY','originZ']
class KSDecayFinder(Algo):
    """
    a class that extracts all the KShorts that have decayed into something
    """

    ## the main 'analysis' method 
    def analyse( self ) :
        """
        The main 'analysis' method
        """
        fout = open(output_fname,'a+')   

        try:

            hdr = self.get("/Event/DAQ/ODIN" )
            eventID= hdr.eventNumber()
            runID= hdr.runNumber()
            #print "now processing run %d, event %d"%(runID,eventID)



            #get Ks
            #k0s = self.select ( 'k0' , 'KS0 -> pi+ pi-' ) #easy way that may not work for old verstions
            # select from input particles according to the decay pattern:
            v0s = self.get ('Rec/Vertex/V0') 
            k0s = self.select ( 'k0' , 'KS0 -> pi+ pi-' )
            
            for v0 in v0s:
                pos = v0.position()
                line = map(str,[runID,eventID,pos.X(),pos.Y(),pos.Z()])
                line = ';'.join(line) +'\n'
                fout.write(line)
       
            if len(k0s)!=0: print "OH GOSH TIS K0S RECO!"      
        except:
            traceback.print_exc()
            self.Print( 'Something went wrong!')
            if not fout.closed:
                fout.close()
                print 'file descriptors closed'
            pass
        return SUCCESS
# =============================================================================

# =============================================================================
## The configuration of the job

def configure ( inputdata , catalogs = [] , castor = False ) :
    
    
    
    from Configurables import DaVinci
    
    DaVinci ( DataType   = '2012',Lumi=False ) 
    
    
    from Configurables import CondDB
    CondDB  ( IgnoreHeartBeat = True )

    
    ## get/create application manager
    gaudi = appMgr() 

    ## define the input data
    setData  ( inputdata , catalogs , castor )
    
    # modify/update the configuration:
    
    # (1) create the algorithm
    alg = KSDecayFinder( 'KsFinder' )
    
    # (2) replace the list of top level algorithm by
    #     new list, which contains only *THIS* algorithm
    gaudi.setAlgorithms( [ alg ] )
    
    return SUCCESS 
# =============================================================================

# =============================================================================
## Job steering 
wdir_derkach = "/afs/cern.ch/work/d/derkach/"
output_fname = "rec_v0_dst05.csv"
if __name__ == '__main__' :

    print 'iBegin!'

    #add csv header
    with open(output_fname,'w') as fout:
        fout.write(';'.join(column_names)+'\n')

    ## job configuration
    inputdata = [
        wdir_derkach+"public/00036954_00000005_1.allstreams.dst"
        #"/afs/cern.ch/work/a/apanin/bg_dst/down/KS_NUNU_md_99.dst"
        ]
    
    configure( inputdata , castor = True )
    
    ## event loop  (number parameter specifies the number of events to process, -1 means process all events)
    run(-1)
    print "iDone!"
        
# =============================================================================
# The END
# =============================================================================



