ctypedef double fftw_complex[2]

cdef class ThreadLocalStorage:
    cdef FFTWPlans plans
    cdef FFTWorkspace work
    cdef public int nfft
    cdef int _initialized
    cpdef void cleanup(self)

cdef class FFTWPlans:
    cdef void* rfft_plan
    cdef void* irfft_plan
    cdef int nfft
    cdef void cleanup(self)
    
cdef class FFTWorkspace:
    cdef double* real_buffer
    cdef fftw_complex* complex_buffer
    cdef double* temp_real        # 临时实数缓冲区
    cdef fftw_complex* temp_complex  # 临时复数缓冲区
    cdef int nfft