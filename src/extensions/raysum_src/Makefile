# Using gfortran on Sun, Mac OS X or FreeBSD: (add -g for debugging)
F77 = gfortran
FFLAGS = -Wall -O3 -g 
LDFLAGS =

EIG = eigenvec.o matrixops.o eispack-cg.o
RAYSUM = raysum.o $(EIG)
IO = readwrite.o
TRACE = phaselist.o buildmodel.o trace.o
MISFIT = misfit.o

# implicit rule for Fortran compilation. Changing the parameter header
# file recompiles everything.
.f.o: params.h
	$(F77) $(FFLAGS) -c $<
        
default: all

all: seis-spread seis-misfit

seis-spread: seis-spread.o $(RAYSUM) $(IO) $(TRACE)
	$(F77) $(LDFLAGS) -o seis-spread seis-spread.o $(RAYSUM) \
                             $(IO) $(TRACE)
	cp seis-spread ../bin
        
seis-misfit: seis-misfit.o $(MISFIT) $(IO)
	$(F77) $(LDFLAGS) -o seis-misfit seis-misfit.o $(MISFIT) $(IO)
	cp seis-misfit ../bin
clean:
	/bin/rm -f *.o *.core seis-spread seis-misfit

