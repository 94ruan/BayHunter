c####&

      program test
      
        implicit none
        include 'params.h'
        
c        character filename*(namelen)
c        integer phaselist(maxseg,2,maxph),nseg(maxph),numph
        
c        filename='../Forward/Sample/sample.ph'
c        call readphases(filename,phaselist,nseg,numph)  
c        call writephases(6,phaselist,nseg,numph)
        
        integer foo
        foo = 3
        write (*,*) foo,foo/2,(foo+1)/2
        
      end
