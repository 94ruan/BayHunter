c Calculates a spread of seismograms for a given model.
c soubroutine provides python itnerface

c####&


c ===================================================================
c Subroutine that only returns traces, phaselist, -amplitudes, -times
c ===================================================================

      subroutine run_full(
     &                    thick, rho, alpha, beta, isoflag, pct,
     &                    trend, plunge, strike, dip, nlay,
     &                    baz, slow, sta_dx, sta_dy, ntr,
     &                    iphase, mults, nsamp, dt, align,
     &                    shift, out_rot, verb, nsegin, numphin,
     &                    phaselistin, Tr_ph, travel_time,
     &                    amplitudeout, phaselist)

c Always start a Fortran program with this line
        implicit none
    
c Include constants:
        include 'params.h'

c ========================================
c Input parameters

c Incoming wave type
        integer iphase
Cf2py intent(in) :: iphase

c Multiple flag, number of samples, trace alignment
c coordinate system of output
        integer mults, nsamp, align
        integer out_rot
Cf2py   integer intent(in) :: mults, nsamp, align
Cf2py   integer intent(in) :: out_rot

c Time step, trace shift 
        real dt, shift
Cf2py   intent(in) :: dt, shift

c model parameters
        integer nlay
        real thick(maxlay), rho(maxlay), alpha(maxlay), beta(maxlay)
        real pct(maxlay), trend(maxlay), plunge(maxlay)
        real strike(maxlay), dip(maxlay)
        logical isoflag(maxlay)
Cf2py   intent(in) :: nlay
Cf2py   intent(in) :: thick, rho, alpha, beta
Cf2py   intent(in) :: pct, trend, plunge
Cf2py   intent(in) :: strike, dip
Cf2py   intent(in) :: isoflag

c Geometry parameters
        real baz(maxtr), slow(maxtr), sta_dx(maxtr), sta_dy(maxtr)
        integer ntr
Cf2py   intent(in) :: ntr
Cf2py   intent(in) :: baz, slow, sta_dx, sta_dy
        integer phaselistin(maxseg,2,maxph)
        integer nsegin(maxph),numphin
Cf2py intent(in) :: nsegin, numphin, phaselistin

c ========================================
c Output parameters
c Traces

        real Tr_ph(3,maxsamp,maxtr)
        integer phaselist(maxseg,2,maxph)
        real travel_time(maxph,maxtr)
        real amplitudeout(3,maxph,maxtr)
Cf2py intent(out) :: Tr_ph, travel_time, amplitudeout
Cf2py intent(out) :: phaselist

c ==================
c Internal variables
c Scratch variables:
        integer j, verb, il, iph, itr
        real amp_in, delta
        logical verbose

c Phase parameters
        real Tr_cart(3,maxsamp,maxtr)
        real amplitude(3,maxph,maxtr)
        integer nseg(maxph), numph

c   aa is a list of rank-4 tensors (a_ijkl = c_ijkl/rho)
c   rot is a list of rotator matrices, used to rotate into the local
c   coordinate system of each interface.
        real aa(3,3,3,3,maxlay), rot(3,3,maxlay)
c   ar_list is aa, pre-rotated into reference frames for the interfaces
c   above (last index=2) and below (1) each respective layer.
        real ar_list(3,3,3,3,maxlay,2)
        
        
c Read in parameters from file 'raysum-params'

        verbose=.false.
        if (verb .eq. 1) then
          verbose=.true.
          print *, 'This is run_full.'
          print *, 'Running verbose.'
        end if

        do il=1,nlay
          if (thick(il) .lt. 0) then
            write (*,*) 'WARNING: Thickness of layer was negative.'
            write (*,*) '         Set to 0.'
            thick(il) = 0
          end if
        end do

c Write out model for testing      
        if (verbose) then
          call writemodel(6,thick,rho,alpha,beta,isoflag,
     &                  pct,trend,plunge,strike,dip,nlay)
        end if
          
c Set up model for calculation, make rotators
        if (verbose) then
          print *, 'Calling buildmodel...'
        end if
        call buildmodel(aa,ar_list,rot,rho,alpha,beta,isoflag,
     &                  pct,trend,plunge,strike,dip,nlay)
     
c Return geometry for testing 
        if (verbose) then
          call writegeom(6,baz,slow,sta_dx,sta_dy,ntr)
        end if
        
c Generate phase list
c Compute direct phases
        if (mults .ne. 3) then
          numph=0
          if (verbose) then
            print *, 'Generating direct phases...'
          end if
          call ph_direct(phaselist,nseg,numph,nlay,iphase)
        end if
c Compute multiples
        if (mults .eq. 1) then
          if (verbose) then
            print *, 'Generating mutiples...'
          end if
          call ph_fsmults(phaselist,nseg,numph,nlay,1,iphase)
          if (verbose) then
            call printphases(phaselist,nseg,numph)
          end if
        else if (mults .eq. 2) then
          if (verbose) then
            print *, 'Generating mutiples...'
          end if
          do j=1,nlay-1
            call ph_fsmults(phaselist,nseg,numph,nlay,j,iphase)
            if (numph .gt. maxph/2) then
               print *, 'Warning: Approaching maximum number of phases!'
               write (*,*) 'Currently: ', numph
               print *, 'Avoid segmentation faults by setting:'
               print *, 'mults to 0, 1, or 3'
            end if
          end do
          if (verbose) then
            call printphases(phaselist,nseg,numph)
          end if
        end if
        
c Perform calculation                   
        if (verbose) then
          print *, 'Calling get_arrivals...'
        end if
        amp_in=1.
        if (mults .eq. 3) then
          if (verbose) then
            print *, 'Using supplied phaselist...'
            call printphases(phaselistin,nsegin,numphin)
          end if
          call get_arrivals(travel_time,amplitude,thick,rho,isoflag,
     &         strike,dip,aa,ar_list,rot,baz,slow,sta_dx,sta_dy,
     &         phaselistin,ntr,nsegin,numphin,nlay,amp_in)
        else
          call get_arrivals(travel_time,amplitude,thick,rho,isoflag,
     &         strike,dip,aa,ar_list,rot,baz,slow,sta_dx,sta_dy,
     &         phaselist,ntr,nseg,numph,nlay,amp_in)
        end if
     
c Normalize arrivals
        if (iphase .eq. 1) then 
          if (verbose) then
            print *, 'Calling norm_arrivals...'
          end if
          if (mults .eq. 3) then
            call norm_arrivals(amplitude,baz,slow,alpha(1),beta(1),
     &                         rho(1),ntr,numphin,1,1)
          else
            call norm_arrivals(amplitude,baz,slow,alpha(1),beta(1),
     &                         rho(1),ntr,numph,1,1)
          end if
        end if
                 
c Assemble traces
        if (verbose) then
          print *, 'Calling make_traces...'
        end if
        if (mults .eq. 3) then
          call make_traces(travel_time,amplitude,ntr,numphin,nsamp,
     &                     dt,align,shift,verbose,Tr_cart)
        else
          call make_traces(travel_time,amplitude,ntr,numph,nsamp,
     &                     dt,align,shift,verbose,Tr_cart)
        end if

        if (out_rot .eq. 0) then
c Write cartesian traces to output
          call copy_traces(Tr_cart,ntr,nsamp,Tr_ph)
          if (mults .eq. 3) then
            call copy_amplitudes(amplitude,ntr,numphin,amplitudeout)
          else
            call copy_amplitudes(amplitude,ntr,numph,amplitudeout)
          end if

        else if (out_rot .eq. 1) then
c Rotate to RTZ
          if (verbose) then
            print *, 'Calling rot_traces...'
          end if
          call rot_traces(Tr_cart,baz,ntr,nsamp,Tr_ph)
          if (mults .eq. 3) then
            call rot_amplitudes(amplitude,baz,ntr,numphin,amplitudeout)
          else
            call rot_amplitudes(amplitude,baz,ntr,numph,amplitudeout)
          end if

        else if (out_rot .eq. 2) then
c   Rotate to wavevector coordinates
            if (verbose) then
              print *, 'Calling fs_traces...'
            end if
            call fs_traces(Tr_cart,baz,slow,alpha(1),beta(1),
     &                     rho(1),ntr,nsamp,Tr_ph)
            if (mults .eq. 3) then
              call fs_amplitudes(amplitude,baz,slow,alpha(1),beta(1),
     &                       rho(1),ntr,numphin,amplitudeout)
            else
              call fs_amplitudes(amplitude,baz,slow,alpha(1),beta(1),
     &                       rho(1),ntr,numph,amplitudeout)
          end if
        end if
                
      end subroutine


c =====================================================
c Subroutine that only returns the traces, no phaselist
c =====================================================
      
      subroutine run_bare(
     &                    thick, rho, alpha, beta, isoflag, pct,
     &                    trend, plunge, strike, dip, nlay,
     &                    baz, slow, sta_dx, sta_dy, ntr,
     &                    iphase, mults, nsamp, dt, align,
     &                    shift, out_rot, verb, nsegin, numphin,
     &                    phaselistin, Tr_ph)

c Always start a Fortran program with this line
        implicit none
    
c Include constants:
        include 'params.h'

c ========================================
c Input parameters

c Incoming wave type
        integer iphase
Cf2py intent(in) :: iphname

c Multiple flag, number of samples, trace alignment
c coordinate system of output
        integer mults, nsamp, align
        integer out_rot
Cf2py   integer intent(in) :: mults, nsamp, align
Cf2py   integer intent(in) :: out_rot

c Time step, trace shift 
        real dt, shift
Cf2py   intent(in) :: dt, shift

c model parameters
        integer nlay
        real thick(maxlay), rho(maxlay), alpha(maxlay), beta(maxlay)
        real pct(maxlay), trend(maxlay), plunge(maxlay)
        real strike(maxlay), dip(maxlay)
        logical isoflag(maxlay)
Cf2py   intent(in) :: nlay
Cf2py   intent(in) :: thick, rho, alpha, beta
Cf2py   intent(in) :: pct, trend, plunge
Cf2py   intent(in) :: strike, dip
Cf2py   intent(in) :: isoflag

c Geometry parameters
        real baz(maxtr), slow(maxtr), sta_dx(maxtr), sta_dy(maxtr)
        integer ntr
Cf2py   intent(in) :: ntr
Cf2py   intent(in) :: baz, slow, sta_dx, sta_dy

        integer phaselistin(maxseg,2,maxph)
        integer nsegin(maxph),numphin
Cf2py intent(in) :: nsegin, numphin, phaselistin

c ========================================
c Output parameters
c Traces

        real Tr_ph(3,maxsamp,maxtr)
Cf2py intent(out) :: Tr_ph

c ==================
c Internal variables
c Scratch variables:
        integer j, verb, il, iph, itr
        real amp_in, delta
        logical verbose

c Phase parameters
        real Tr_cart(3,maxsamp,maxtr)
        integer phaselist(maxseg,2,maxph)
        real travel_time(maxph,maxtr)
        real amplitude(3,maxph,maxtr)
        integer nseg(maxph), numph

c   aa is a list of rank-4 tensors (a_ijkl = c_ijkl/rho)
c   rot is a list of rotator matrices, used to rotate into the local
c   coordinate system of each interface.
        real aa(3,3,3,3,maxlay), rot(3,3,maxlay)
c   ar_list is aa, pre-rotated into reference frames for the interfaces
c   above (last index=2) and below (1) each respective layer.
        real ar_list(3,3,3,3,maxlay,2)
        
        
c Read in parameters from file 'raysum-params'

        verbose=.false.
        if (verb .eq. 1) then
          verbose=.true.
          print *, 'This is run_bare.'
          print *, 'Running verbose.'
        end if

        do il=1,nlay
          if (thick(il) .lt. 0) then
            write (*,*) 'WARNING: Thickness of layer was negative.'
            write (*,*) '         Set to 0.'
            thick(il) = 0
          end if
        end do

c Set up model for calculation, make rotators
        if (verbose) then
          print *, 'Calling buildmodel...'
        end if
        call buildmodel(aa,ar_list,rot,rho,alpha,beta,isoflag,
     &                  pct,trend,plunge,strike,dip,nlay)
     
c Generate phase list
c Compute direct phases
        if (mults .ne. 3) then
          numph=0
          if (verbose) then
            print *, 'Generating direct phases...'
          end if
          call ph_direct(phaselist,nseg,numph,nlay,iphase)
        end if
c Compute multiples
        if (mults .eq. 1) then
          if (verbose) then
            print *, 'Generating mutiples...'
          end if
          call ph_fsmults(phaselist,nseg,numph,nlay,1,iphase)
          if (verbose) then
            call printphases(phaselist,nseg,numph)
          end if
        else if (mults .eq. 2) then
          if (verbose) then
            print *, 'Generating mutiples...'
          end if
          do j=1,nlay-1
            call ph_fsmults(phaselist,nseg,numph,nlay,j,iphase)
          end do
          if (verbose) then
            call printphases(phaselist,nseg,numph)
          end if
        end if
        
c Perform calculation                   
        if (verbose) then
          print *, 'Calling get_arrivals...'
        end if
        amp_in=1.
        if (mults .eq. 3) then
          if (verbose) then
            print *, 'Using supplied phaselist...'
            call printphases(phaselistin,nsegin,numphin)
          end if
          call get_arrivals(travel_time,amplitude,thick,rho,isoflag,
     &         strike,dip,aa,ar_list,rot,baz,slow,sta_dx,sta_dy,
     &         phaselistin,ntr,nsegin,numphin,nlay,amp_in)
        else
          call get_arrivals(travel_time,amplitude,thick,rho,isoflag,
     &         strike,dip,aa,ar_list,rot,baz,slow,sta_dx,sta_dy,
     &         phaselist,ntr,nseg,numph,nlay,amp_in)
        end if
     
c Normalize arrivals
        if (iphase .eq. 1) then 
          if (verbose) then
            print *, 'Calling norm_arrivals...'
          end if
          if (mults .eq. 3) then
            call norm_arrivals(amplitude,baz,slow,alpha(1),beta(1),
     &                         rho(1),ntr,numphin,1,1)
          else
            call norm_arrivals(amplitude,baz,slow,alpha(1),beta(1),
     &                         rho(1),ntr,numph,1,1)
          end if
        end if
                 
c Assemble traces
        if (verbose) then
          print *, 'Calling make_traces...'
        end if
        if (mults .eq. 3) then
          call make_traces(travel_time,amplitude,ntr,numphin,nsamp,
     &                     dt,align,shift,verbose,Tr_cart)
        else
          call make_traces(travel_time,amplitude,ntr,numph,nsamp,
     &                     dt,align,shift,verbose,Tr_cart)
        end if

        if (out_rot .eq. 0) then
c Write cartesian traces to output
          call copy_traces(Tr_cart,ntr,nsamp,Tr_ph)

        else if (out_rot .eq. 1) then
c Rotate to RTZ
          if (verbose) then
            print *, 'Calling rot_traces...'
          end if
          call rot_traces(Tr_cart,baz,ntr,nsamp,Tr_ph)

        else if (out_rot .eq. 2) then
c   Rotate to wavevector coordinates
            if (verbose) then
              print *, 'Calling fs_traces...'
            end if
            call fs_traces(Tr_cart,baz,slow,alpha(1),beta(1),
     &                     rho(1),ntr,nsamp,Tr_ph)
        end if
                
      end subroutine

c =====================================================
c Subroutine For MCMC
c =====================================================

      subroutine run_bare_mcmc(thick, rho, alpha, beta, isoflag, pct,
     &    trend, plunge, strike, dip, nlay, baz, slow, sta_dx,
     &    sta_dy, ntr, iphase, mults, nsamp, dt, align, shift,
     &    out_rot, verb, nsegin, numphin, phaselistin, traces)

      implicit none
      include 'params.h'

c ========================================
c Input parameters (annotated with Cf2py directives)
c ========================================

c Layer properties
      real thick(maxlay), rho(maxlay), alpha(maxlay), beta(maxlay)
      real pct(maxlay), trend(maxlay), plunge(maxlay)
      real strike(maxlay), dip(maxlay)

Cf2py intent(in) :: thick, rho, alpha, beta, pct, trend, plunge
Cf2py intent(in) :: strike, dip
Cf2py depend(nlay) :: thick, rho, alpha, beta, pct, trend, plunge
Cf2py depend(nlay) :: strike, dip

c Isotropic flag
      logical isoflag(maxlay)
Cf2py intent(in) :: isoflag
Cf2py depend(nlay) :: isoflag

c Station and ray parameters
      real baz(maxtr), slow(maxtr), sta_dx(maxtr), sta_dy(maxtr)
Cf2py intent(in) :: baz, slow, sta_dx, sta_dy
Cf2py depend(ntr) :: baz, slow, sta_dx, sta_dy

c Control parameters
      integer nlay, ntr, iphase, mults, nsamp, align, out_rot
Cf2py intent(in) :: nlay, ntr, iphase, mults, nsamp, align, out_rot

c Time step and shift
      real dt, shift
Cf2py intent(in) :: dt, shift

c Verbosity and phase list
      integer verb, nsegin(maxph), numphin
      integer phaselistin(maxseg,2,maxph)
Cf2py intent(in), optional :: verb, nsegin, numphin, phaselistin
Cf2py :: verb = 0
Cf2py :: nsegin = 0
Cf2py :: numphin = 0
Cf2py :: phaselistin = 0


c Output traces
      real traces(3, maxsamp, maxtr)
Cf2py intent(inout) :: traces
Cf2py depend(nsamp, ntr) :: traces

c ========================================
c Local variables
c ========================================
      external estimate_tt
      real aa(3,3,3,3,maxlay), rot(3,3,maxlay)
      real ar_list(3,3,3,3,maxlay,2)
      real Tr_cart(3,maxsamp,maxtr)
      real travel_time(maxph,maxtr)
      real amplitude(3,maxph,maxtr)
      real amp_in
      integer numph, nseg, j
      logical verbose

c Initialize verbosity
      verbose=.false.
        if (verb .eq. 1) then
          verbose=.true.
          print *, 'This is run_bare.'
          print *, 'Running verbose.'
        end if

c Default amplitude scaling
      amp_in = 1.0
      
c Build the model
      call buildmodel(aa, ar_list, rot, rho, alpha, beta, isoflag,
     &    pct, trend, plunge, strike, dip, nlay)

c Generate phase list if not provided
      if (mults .ne. 3) then
          numph = 0
          if (verbose) print *, 'Generating direct phases...'
          call ph_direct(phaselistin, nsegin(1), numph, nlay, iphase)
          call filter_phases(phaselistin, nsegin, numph, thick, alpha,
     &        beta)
      endif

c Generate multiples if requested
      if (mults .eq. 1) then
          if (verbose) print *, 'Generating multiples...'
          call ph_fsmults(phaselistin, nsegin(1), numph,
     &        nlay, 1, iphase)
          call filter_phases(phaselistin, nsegin, numph, thick, alpha,
     &        beta)
          if (verbose) call printphases(phaselistin, nsegin(1), numph)
      else if (mults .eq. 2) then
          if (verbose) print *, 'Generating multiples...'
          do j = 1, nlay-1
              call ph_fsmults(phaselistin, nsegin(1), numph,
     &            nlay, j, iphase)
          enddo
          call filter_phases(phaselistin, nsegin, numph, thick, alpha,
     &        beta)
          if (verbose) call printphases(phaselistin, nsegin(1), numph)
      endif

c Compute travel times and amplitudes
      if (verbose) print *, 'Calling get_arrivals...'
      if (mults .eq. 3) then
          if (verbose) then
              print *, 'Using supplied phaselist...'
              call printphases(phaselistin, nsegin, numphin)
          endif
          call get_arrivals(travel_time, amplitude, thick, rho,
     &        isoflag, strike, dip, aa, ar_list, rot, baz, slow,
     &        sta_dx, sta_dy, phaselistin, ntr, nsegin, numphin,
     &        nlay, amp_in)
      else
          call get_arrivals(travel_time, amplitude, thick, rho,
     &        isoflag, strike, dip, aa, ar_list, rot, baz, slow,
     &        sta_dx, sta_dy, phaselistin, ntr, nsegin, numph,
     &        nlay, amp_in)
      endif

c Normalize amplitudes if needed
      if (iphase .eq. 1) then
          if (verbose) print *, 'Calling norm_arrivals...'
          if (mults .eq. 3) then
              call norm_arrivals(amplitude, baz, slow, alpha(1),
     &            beta(1), rho(1), ntr, numphin, 1, 1)
          else
              call norm_arrivals(amplitude, baz, slow, alpha(1),
     &            beta(1), rho(1), ntr, numph, 1, 1)
          endif
      endif

c Generate synthetic traces
      if (verbose) print *, 'Calling make_traces...'
      if (mults .eq. 3) then
          call make_traces(travel_time, amplitude, ntr, numphin,
     &        nsamp, dt, align, shift, verbose, Tr_cart)
      else
          call make_traces(travel_time, amplitude, ntr, numph,
     &        nsamp, dt, align, shift, verbose, Tr_cart)
      endif

c Rotate traces if requested
      if (out_rot .eq. 0) then
          traces = Tr_cart
      else if (out_rot .eq. 1) then
          call rot_traces(Tr_cart, baz, ntr, nsamp, traces)
      else if (out_rot .eq. 2) then
          call fs_traces(Tr_cart, baz, slow, alpha(1), beta(1),
     &        rho(1), ntr, nsamp, traces)
      endif

      end subroutine

      real function estimate_tt(phaselist, nseg, thick, alpha, beta)
c ==========================================================
c   Estimate total travel time for one phase path, considering slow
c ==========================================================
      implicit none
c Include constants:
      include 'params.h'
      
      integer phaselist(maxseg,2)
      integer nseg
      real thick(maxlay),alpha(maxlay),beta(maxlay)
      integer i,lay,wavetype
      real vel,slow,vsq_inv,usq

      estimate_tt = 0.0
      slow = 0.06 / 1000.0
      usq = slow*slow

      do i = 1,nseg
          lay = phaselist(i,1)
          wavetype = phaselist(i,2)
          if (wavetype.eq.1) then
              vel = alpha(lay)
          else if (wavetype.eq.2) then
              vel = beta(lay)
          else
              vel = alpha(lay)
          endif
          vsq_inv = 1.0/(vel*vel)
          if (vsq_inv.gt.usq) then
              estimate_tt = estimate_tt + thick(lay)*
     &                     sqrt(vsq_inv-usq)
          else
          endif
      end do

      return
      end

      
      subroutine filter_phases(phaselist, nsegin, numph, thick, alpha, 
     &          beta)
c ==========================================================
c   Filter phase list to keep only phases with travel time < tmax
c ==========================================================
      implicit none
c Include constants:
      include 'params.h'   
      real estimate_tt
      integer phaselist(maxseg,2,maxph)
      integer nsegin(maxph)
      integer numph
      real thick(maxlay), alpha(maxlay), beta(maxlay)
      integer i, keep
      real ttime
      real tmax

      tmax = 10.0
      keep = 0

      do i = 1, numph
          ttime = estimate_tt(phaselist(1,1,i), nsegin(i), thick, alpha, 
     &    beta)
          if (ttime .gt. 0.0 .and. ttime .lt. tmax) then
              keep = keep + 1
c Copy phase i to new position keep
              call copy_phase(phaselist(1,1,i), nsegin(i), 
     &                       phaselist(1,1,keep), nsegin(keep))
          endif
      end do

      numph = keep

      return
      end

      subroutine copy_phase(phase_in, nseg_in, phase_out, nseg_out)
c ==========================================================
c   Copy one phase (phaselist) to another
c ==========================================================
      implicit none
c Include constants:
      include 'params.h'      
      integer phase_in(maxseg,2), phase_out(maxseg,2)
      integer nseg_in, nseg_out
      integer i

      nseg_out = nseg_in

      do i = 1, nseg_in
          phase_out(i,1) = phase_in(i,1)
          phase_out(i,2) = phase_in(i,2)
      end do

      return
      end
