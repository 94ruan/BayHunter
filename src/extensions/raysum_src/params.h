c####&
c namelen is the length of filenames (in characters)
c maxlay is the maximum number of layers allowed in the model
c maxtr is the maximum number of traces allowed
c maxseg: maximum # of segments (should be 3*maxlay for 1st-order
c         multiples
c maxph: maximum number of phases per trace (needs to be LARGE
c        if using many multiples)
c buffsize is the max. line length assumed for reading files.
      integer namelen, maxlay, maxtr, maxseg, maxph, buffsize
      parameter (namelen=40,maxlay=12,maxtr=300,maxseg=36)
      parameter (maxph=50000,buffsize=256)
      
c Units for reading and writing
      integer iounit1,iounit2
      parameter (iounit1=1,iounit2=2)
      
c pi: duh. ztol: tolerance considered to be equiv. to zero.
      real pi,ztol
      parameter (pi=3.141592653589793,ztol=1.e-7)
      
c maxsamp is the max. number of samples per trace.
      integer maxsamp
      parameter (maxsamp=6000)
