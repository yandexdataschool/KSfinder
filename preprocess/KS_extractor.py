''#!/usr/bin/env python
from Bender.Main import *
# =============================================================================
#whatdoido: i process damn monte-carlo to get KS->something decays
import traceback
column_names = ['runID','eventID','ks_id','endVertex_id','children',
                'originX','originY','originZ',
                'primaryX','primaryY','primaryZ',
                'decayX','decayY','decayZ']
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
            print "now processing run %d, event %d"%(runID,eventID)



            #get Ks
            #k0s = self.select ( 'k0' , 'KS0 -> pi+ pi-' ) #easy way that may not work for old verstions

            mcps= list(self.get("MC/Particles"))      #all particles


            ks_code = 310
            pi_code = 211
            pi0_code = 111     

            ks_particles = filter(lambda p:p.particleID().pid()==ks_code, mcps)
                
                
            #Iterate over Ks particles to find decays
            for p_i,p in enumerate(ks_particles): 
                

                originVertex = p.originVertex()
                origin = originVertex.position()

                primaryVertex = p.primaryVertex()
                primary = primaryVertex.position()

                endVertices = p.endVertices()
                for v_i in range(len(endVertices)):
                    endv = endVertices[v_i].target()
                    products =  endv.products()
                    
                    #children PARTICLES
                    children = [prod.target() for prod in products]
                    decay = endv.position()
                                
                    row = [runID,eventID,p_i,v_i,
                            '&'.join([str(child.particleID().pid()) for child in children]),
                            origin.X(),origin.Y(),origin.Z(),
                            primary.X(),primary.Y(),primary.Z(),
                            decay.X(),decay.Y(),decay.Z()]

                    csv_line = ';'.join(map(str,row) )
                    fout.write(csv_line+"\n")
                                
                if len(endVertices)==0:
                    row = [runID,eventID,-1,-1,
                            'no_decay',
                            'nan','nan','nan',
                            'nan','nan','nan',
                            'nan','nan','nan',]

                    csv_line = ';'.join(map(str,row) )
                    fout.write(csv_line+"\n")       
             

            fout.close()

            print len(ks_particles),'K0s found'    
              
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
output_fname = "KS_decays_6.csv"
if __name__ == '__main__' :

    print 'iBegin!'

    #add csv header
    with open(output_fname,'w') as fout:
        fout.write(';'.join(column_names)+'\n')

    ## job configuration
    inputdata = [
        wdir_derkach+"public/00036954_00000006_1.allstreams.dst"
        #"/afs/cern.ch/work/a/apanin/bg_dst/down/KS_NUNU_md_99.dst"
        ]
    
    configure( inputdata , castor = True )
    
    ## event loop  (number parameter specifies the number of events to process, -1 means process all events)
    run(-1)
    print "iDone!"
        
# =============================================================================
# The END
# =============================================================================



