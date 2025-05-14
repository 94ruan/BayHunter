# cython: language_level=3
# distutils: extra_compile_args = -ffast-math -O3 -march=native
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

#----------------------------------------------------------Version Division----------------------------------------------------------
import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, cos, sin, fabs, pow, M_PI
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t
# from scipy.fftpack import fft, ifft
from cython.parallel import prange
from libc.stdio cimport printf

np.import_array()

# Type definitions
DTYPE = np.float64
CTYPE = np.complex128

ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CTYPE_t

cdef extern from "alloca.h":
    void* alloca(size_t size) nogil

cdef extern from "math.h":
    double round(double x) noexcept nogil  # C版本的round函数
    double cos(double x) noexcept nogil
    double sin(double x) noexcept nogil
    double sqrt(double x) noexcept nogil
    
cdef extern from "fftw3.h":
    ctypedef double fftw_complex[2]
    void fftw_init_threads()
    void fftw_plan_with_nthreads(int nthreads)
    void* fftw_plan_dft_r2c_1d(int n, double* in_arr, fftw_complex* out_arr, unsigned flags) noexcept
    void* fftw_plan_dft_c2r_1d(int n, fftw_complex* in_arr, double* out_arr, unsigned flags) noexcept
    void fftw_execute_dft_r2c(void* plan, double* in_arr, fftw_complex* out_arr) noexcept nogil
    void fftw_execute_dft_c2r(void* plan, fftw_complex* in_arr, double* out_arr) noexcept nogil
    void fftw_make_planner_thread_safe()
    void fftw_destroy_plan(void* plan) noexcept
    void* fftw_malloc(size_t size) noexcept
    void fftw_free(void* ptr) noexcept
    void fftw_cleanup_threads()
    void fftw_cleanup()
    
    unsigned FFTW_ESTIMATE
    unsigned FFTW_MEASURE
    unsigned FFTW_PATIENT
    unsigned FFTW_EXHAUSTIVE

cdef class ThreadLocalStorage:
    cdef FFTWPlans plans
    cdef FFTWorkspace work
    cdef public int nfft
    cdef int _initialized
    cpdef void cleanup(self)
    
    def __cinit__(self, int nfft):
        self.nfft = nfft
        self._initialized = 0

        # 线程安全初始化（仅首次）
        if not self._initialized:
            fftw_init_threads()
            fftw_make_planner_thread_safe()
            self._initialized = 1

        fftw_plan_with_nthreads(1)  # 每个计划单线程执行
        self.plans = FFTWPlans(nfft)
        self.work = FFTWorkspace(nfft)

    cpdef void cleanup(self):
        """轻量级清理：仅重建计划，保留工作内存"""
        self.plans.cleanup()
        self.plans = FFTWPlans(self.nfft)

    def __dealloc__(self):
        self.plans.cleanup()

cdef class FFTWPlans:
    cdef void* rfft_plan
    cdef void* irfft_plan
    cdef int nfft
    cdef void cleanup(self)
    
    def __cinit__(self, int nfft):
        self.nfft = nfft
        # 使用临时变量创建计划后立即释放
        in_real = <double*>fftw_malloc(nfft * sizeof(double))
        out_cplx = <fftw_complex*>fftw_malloc((nfft//2+1) * sizeof(fftw_complex))
        self.rfft_plan = fftw_plan_dft_r2c_1d(nfft, in_real, out_cplx, FFTW_ESTIMATE)
        self.irfft_plan = fftw_plan_dft_c2r_1d(nfft, out_cplx, in_real, FFTW_ESTIMATE)
        fftw_free(in_real)
        fftw_free(out_cplx)
    
    cdef void cleanup(self):
        """简化清理逻辑"""
        if self.rfft_plan != NULL:
            fftw_destroy_plan(self.rfft_plan)
            self.rfft_plan = NULL
        if self.irfft_plan != NULL:
            fftw_destroy_plan(self.irfft_plan)
            self.irfft_plan = NULL
        
cdef class FFTWorkspace:
    cdef double* real_buffer
    cdef fftw_complex* complex_buffer
    cdef double* temp_real        # 临时实数缓冲区
    cdef fftw_complex* temp_complex  # 临时复数缓冲区
    cdef int nfft
    
    def __cinit__(self, int nfft):
        self.nfft = nfft
        self.real_buffer = <double*>fftw_malloc(nfft * sizeof(double))
        self.complex_buffer = <fftw_complex*>fftw_malloc((nfft//2+1) * sizeof(fftw_complex))
        self.temp_real = <double*>fftw_malloc(nfft * sizeof(double))
        self.temp_complex = <fftw_complex*>fftw_malloc(nfft * sizeof(fftw_complex))
        
    def __dealloc__(self):
        fftw_free(self.real_buffer)
        fftw_free(self.complex_buffer)
        fftw_free(self.temp_real)
        fftw_free(self.temp_complex)
        
# 修改后的实数FFT函数（保持输出长度为nfft）
cdef void rfft_fftw_reuse(FFTWPlans plans, FFTWorkspace work, 
                         DTYPE_t* in_real, CTYPE_t* out_complex) noexcept nogil:
    # 直接使用输入指针，避免额外memcpy
    fftw_execute_dft_r2c(plans.rfft_plan, in_real, work.complex_buffer)
    
    # 更高效的内存布局处理
    cdef int i, half = plans.nfft // 2
    cdef double[2]* complex_view = <double[2]*>work.complex_buffer
    cdef double[2]* out_view = <double[2]*>out_complex
    
    # 处理0频率分量
    out_view[0][0] = complex_view[0][0]
    out_view[0][1] = complex_view[0][1]
    
    # 处理正频率和对应的负频率
    for i in range(1, half + 1):
        out_view[i][0] = complex_view[i][0]
        out_view[i][1] = complex_view[i][1]
        if i < half:
            out_view[plans.nfft-i][0] = complex_view[i][0]
            out_view[plans.nfft-i][1] = -complex_view[i][1]
    
    # 处理Nyquist频率(当N为偶数时)
    if plans.nfft % 2 == 0:
        out_complex[half].imag = 0.0

cdef void irfft_fftw_reuse(FFTWPlans plans, FFTWorkspace work,
                          CTYPE_t* in_complex, DTYPE_t* out_real) noexcept nogil:
    # 直接准备输入数据到工作缓冲区
    cdef double[2]* complex_view = <double[2]*>work.complex_buffer
    cdef int i, half = plans.nfft // 2
    
    # 0频率分量
    complex_view[0][0] = in_complex[0].real
    complex_view[0][1] = in_complex[0].imag
    
    # 正频率分量
    for i in range(1, half + 1):
        complex_view[i][0] = in_complex[i].real
        complex_view[i][1] = in_complex[i].imag
    
    # 执行变换
    fftw_execute_dft_c2r(plans.irfft_plan, work.complex_buffer, out_real)
    
    # 就地归一化
    cdef double norm = 1.0 / plans.nfft
    for i in range(plans.nfft):
        out_real[i] *= norm
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterdecon(np.ndarray[DTYPE_t, ndim=3] traces_in, 
              np.ndarray[DTYPE_t, ndim=1] baz, 
              int nfft, 
              np.ndarray[CTYPE_t, ndim=1] gaussF, 
#               np.ndarray[CTYPE_t, ndim=1] gaussF_nor, 
              int odina_flag,
              DTYPE_t tshift=1, 
              int itmax=200, 
              DTYPE_t minderr=0.01, 
              bint use_bic=True, 
              DTYPE_t dt_bare=1,
              ThreadLocalStorage tls=None,
              int nused=73):
    cdef np.ndarray[DTYPE_t, ndim=3] traces = np.empty((nused, 3, nfft), dtype=DTYPE)
    cdef int i, j, k
    for i in range(3):
        for j in range(nused):
            for k in range(nfft):
                traces[j, i, k] = traces_in[i, k, j]
#     cdef bint own_tls = False
#     if tls is None:
#         tls = ThreadLocalStorage(nfft)
#         own_tls = True
        
    cdef int num_traces = traces.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] forward_list = np.zeros((num_traces, 2, nfft), dtype=DTYPE)
    
    # 使用连续内存视图
    cdef DTYPE_t[:, :, ::1] forward_view = forward_list
    cdef DTYPE_t[:, :, ::1] traces_view = traces
    
    # 线程本地存储初始化
#     cdef ThreadLocalStorage tls = ThreadLocalStorage(nfft)
    cdef FFTWPlans plans = tls.plans
    cdef FFTWorkspace work = tls.work
    
#     printf("[iterdecon] tls plan_f addr: %p | plan_b addr: %p | nfft: %d\n", 
#        <void *>tls.plan_f, <void *>tls.plan_b, tls.nfft)
    
    for i in prange(num_traces, nogil=True, schedule='dynamic'):
        # 直接操作内存视图
        forward_rf_parallel(
            plans, work,
            &traces_view[i, 0, 0],  # 直接传递第i个trace的指针
            nfft,
            &gaussF[0],
#             &gaussF_nor[0],
            odina_flag,
            tshift, itmax, minderr,
            use_bic, dt_bare,
            &forward_view[i, 0, 0]  # 直接写入结果数组
        )
    
#     if own_tls:
#         del tls  # 保证释放本地创建的资源
        
    return forward_list

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void forward_rf_parallel(FFTWPlans plans, 
                            FFTWorkspace work,
                            DTYPE_t* stream_ptr, 
                            int nfft,
                            CTYPE_t* gaussF_ptr,
                            int odina_flag,
                            DTYPE_t tshift, 
                            int itmax, 
                            DTYPE_t minderr, 
                            bint use_bic, 
                            DTYPE_t dt_bare,
                            DTYPE_t* output_ptr) noexcept nogil:
    
    # 声明所有变量
    cdef int half = nfft // 2
    cdef int index, chan, it, index_k, shift_i, final_index, rf_index, non_zero, j
    cdef DTYPE_t amp, powerU, powerR, sumsq, sumsq_i, d_error, max_val, powerW, temp_re, temp_im
    cdef DTYPE_t factor = dt_bare * dt_bare
    cdef double norm, diff
    cdef int min_index = 0
        
    # 使用栈内存避免线程竞争
    cdef DTYPE_t* W0 = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef DTYPE_t* U = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef DTYPE_t* P0 = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef DTYPE_t* P = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef DTYPE_t* temp_real = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef DTYPE_t* rf = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    
    cdef CTYPE_t* W0f = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
    cdef CTYPE_t* Wf = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
    cdef CTYPE_t* temp_complex = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
    cdef CTYPE_t* phase_shift = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
    
    cdef DTYPE_t* temp_rf = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    cdef double current_bic, min_bic
    cdef bint has_solution = False
    
    # 初始化W0
    if odina_flag == 2:
        for j in range(nfft):
            W0[j] = stream_ptr[j]  # 第一通道
    else:
        for j in range(nfft):
            W0[j] = stream_ptr[2*nfft + j]  # 第三通道
    
    # FFT运算
    rfft_fftw_reuse(plans, work, W0, W0f)
    
    # 频域滤波
    for j in range(nfft): #(与Python版本一致: Wf = W0f * gaussF * dt_bare)
        Wf[j].real = (W0f[j].real * gaussF_ptr[j].real - W0f[j].imag * gaussF_ptr[j].imag) * dt_bare
        Wf[j].imag = (W0f[j].real * gaussF_ptr[j].imag + W0f[j].imag * gaussF_ptr[j].real) * dt_bare
    
    # IFFT运算
    irfft_fftw_reuse(plans, work, Wf, temp_real)
    
    # 计算powerW
    powerW = 0.0
    for j in range(nfft):
        powerW += temp_real[j] * temp_real[j]
    cdef double inv_powerW = 1.0 / powerW
    
    # 处理两个分量（径向和切向）
    for index in range(2):
        has_solution = False
        current_bic = 0.0
        min_bic = 1e20
        for j in range(nfft):
            temp_rf[j] = 0.0
        # 确定通道索引
        chan = 1 if (odina_flag == 2 and index == 0) else (2 if (odina_flag == 2) else index)
        
        # 初始化U
        for j in range(nfft):
            U[j] = stream_ptr[chan*nfft + j]
        
        # FFT运算
        rfft_fftw_reuse(plans, work, U, temp_complex)
        
        # 频域滤波
        for j in range(nfft):
            temp_re = temp_complex[j].real * gaussF_ptr[j].real - temp_complex[j].imag * gaussF_ptr[j].imag
            temp_im = temp_complex[j].real * gaussF_ptr[j].imag + temp_complex[j].imag * gaussF_ptr[j].real
            temp_complex[j].real = temp_re * dt_bare
            temp_complex[j].imag = temp_im * dt_bare
        
        # IFFT运算
        irfft_fftw_reuse(plans, work, temp_complex, U)
        
        # 计算powerU
        powerU = 0.0
        for j in range(nfft):
            powerU += U[j] * U[j]
        
        # 初始化P0和temp_real (作为残差R)
        for j in range(nfft):
            P0[j] = 0.0
            temp_real[j] = U[j]
        
        sumsq_i = 1.0
        d_error = 100.0 * powerU + minderr
        
        # 开始迭代反卷积
        for it in range(itmax):
            if it == 0 and fabs(d_error) == minderr:
                # 初始解处理
                for j in range(nfft):
                    temp_rf[j] = P0[j]
                min_bic = log(nfft) * 2 + nfft * log(powerU / nfft)
                has_solution = True
                break
            elif fabs(d_error) > minderr:
                # FFT卷积运算
                rfft_fftw_reuse(plans, work, temp_real, temp_complex)
                
                # 复数乘法 (A * conj(B))
                for j in range(nfft):
                    temp_re = (temp_complex[j].real * Wf[j].real + temp_complex[j].imag * Wf[j].imag)
                    temp_im = (temp_complex[j].imag * Wf[j].real - temp_complex[j].real * Wf[j].imag)
                    temp_complex[j].real = temp_re * inv_powerW
                    temp_complex[j].imag = temp_im * inv_powerW
                
                # IFFT运算
                irfft_fftw_reuse(plans, work, temp_complex, temp_real)
                
                # 寻找最大值
                index_k = 0
                max_val = 0.0
                for j in range(half):
                    if fabs(temp_real[j]) > max_val:
                        max_val = fabs(temp_real[j])
                        index_k = j
                
                amp = temp_real[index_k] / dt_bare
                P0[index_k] += amp
                
                # 计算预测值
                rfft_fftw_reuse(plans, work, P0, temp_complex)
                
                # 频域滤波
                for j in range(nfft):
                    creal = temp_complex[j].real
                    cimag = temp_complex[j].imag
                    # 复数乘法: (gaussF * W0f * dt_bare^2)
                    filter_real = gaussF_ptr[j].real * W0f[j].real - gaussF_ptr[j].imag * W0f[j].imag
                    filter_imag = gaussF_ptr[j].real * W0f[j].imag + gaussF_ptr[j].imag * W0f[j].real
                    temp_complex[j].real = (creal * filter_real - cimag * filter_imag) * factor
                    temp_complex[j].imag = (creal * filter_imag + cimag * filter_real) * factor
                
                # IFFT运算
                irfft_fftw_reuse(plans, work, temp_complex, P)
                
                # 计算残差
                powerR = 0.0
                non_zero = 0
                for j in range(nfft):
                    diff = U[j] - P[j]
                    temp_real[j] = diff
                    powerR += diff * diff
                    if P0[j] != 0.0:
                        non_zero += 1
                        
                sumsq = powerR / powerU
                
                # 计算当前BIC
                current_bic = log(nfft) * non_zero * 2 + nfft * log(powerR / nfft)
                # 更新最优解
                if current_bic < min_bic:
                    min_bic = current_bic
                    for j in range(nfft):
                        temp_rf[j] = P0[j]
                    has_solution = True

                # 更新误差
                d_error = 100.0 * (sumsq_i - sumsq)
                sumsq_i = sumsq
            else:
                break
        
        # 复制最优解到输出
        if has_solution:
            for j in range(nfft):
                rf[j] = temp_rf[j]
        else:
            for j in range(nfft):
                rf[j] = 0.0
        
        # FFT运算
        rfft_fftw_reuse(plans, work, rf, temp_complex)
        
        # 频域滤波
        for j in range(nfft):
            temp_re = temp_complex[j].real * gaussF_ptr[j].real - temp_complex[j].imag * gaussF_ptr[j].imag
            temp_im = temp_complex[j].real * gaussF_ptr[j].imag + temp_complex[j].imag * gaussF_ptr[j].real
            temp_complex[j].real = temp_re
            temp_complex[j].imag = temp_im
        
        # IFFT运算
        irfft_fftw_reuse(plans, work, temp_complex, rf)
        
        # 相位校正
        shift_i = <int>(round(tshift / dt_bare))
        for j in range(nfft):
            phase_shift[j].real = cos(2 * M_PI * j * shift_i / nfft)
            phase_shift[j].imag = -sin(2 * M_PI * j * shift_i / nfft)
        
        # FFT运算
        rfft_fftw_reuse(plans, work, rf, temp_complex)
        
        # 相位校正
        for j in range(nfft):
            temp_re = temp_complex[j].real * phase_shift[j].real - temp_complex[j].imag * phase_shift[j].imag
            temp_im = temp_complex[j].real * phase_shift[j].imag + temp_complex[j].imag * phase_shift[j].real
            temp_complex[j].real = temp_re
            temp_complex[j].imag = temp_im
        
        # IFFT运算
        irfft_fftw_reuse(plans, work, temp_complex, rf)
        
        # 归一化
        norm = 1.0 / (cos(2 * M_PI * shift_i / nfft) + 1e-10)  # 避免除以零
        for j in range(nfft):
            rf[j] *= norm
        
        # 存储结果
        for j in range(nfft):
            output_ptr[index*nfft + j] = rf[j]
#----------------------------------------------------------Version Division----------------------------------------------------------
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef void forward_rf_parallel(FFTWPlans plans, 
#                             FFTWorkspace work,
#                             DTYPE_t* stream_ptr, 
#                             int nfft,
#                             CTYPE_t* gaussF_ptr,
#                             CTYPE_t* gaussF_nor_ptr,
#                             int odina_flag,
#                             DTYPE_t tshift, 
#                             int itmax, 
#                             DTYPE_t minderr, 
#                             bint use_bic, 
#                             DTYPE_t dt_bare,
#                             DTYPE_t* output_ptr) noexcept nogil:
    
#     # 声明所有变量
#     cdef int half = nfft // 2
#     cdef int index, chan, it, index_k, shift_i, final_index, rf_index, non_zero, j
#     cdef DTYPE_t amp, powerU, powerR, sumsq, sumsq_i, d_error, max_val, powerW, temp_re, temp_im, 
#     cdef DTYPE_t factor = dt_bare * dt_bare
#     cdef double norm, diff
#     cdef int min_index = 0
        
#     # 使用栈内存避免线程竞争
#     cdef DTYPE_t* W0 = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef DTYPE_t* U = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef DTYPE_t* P0 = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef DTYPE_t* P = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef DTYPE_t* temp_real = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef DTYPE_t* rf = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
    
#     cdef CTYPE_t* W0f = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
#     cdef CTYPE_t* Wf = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
#     cdef CTYPE_t* temp_complex = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
#     cdef CTYPE_t* phase_shift = <CTYPE_t*>alloca(nfft * sizeof(CTYPE_t))
    
#     cdef DTYPE_t* temp_rf = <DTYPE_t*>alloca(nfft * sizeof(DTYPE_t))
#     cdef double current_bic, min_bic
#     cdef bint has_solution = False
    
#     # 初始化W0
#     if odina_flag == 2:
#         for j in range(nfft):
#             W0[j] = stream_ptr[j]  # 第一通道
#     else:
#         for j in range(nfft):
#             W0[j] = stream_ptr[2*nfft + j]  # 第三通道
    
#     # FFT运算
#     rfft_fftw_reuse(plans, work, W0, W0f)
    
#     # 频域滤波
#     for j in range(nfft): #(与Python版本一致: Wf = W0f * gaussF * dt_bare)
#         Wf[j].real = (W0f[j].real * gaussF_ptr[j].real - W0f[j].imag * gaussF_ptr[j].imag)
#         Wf[j].imag = (W0f[j].real * gaussF_ptr[j].imag + W0f[j].imag * gaussF_ptr[j].real)
    
#     # IFFT运算
#     irfft_fftw_reuse(plans, work, Wf, temp_real)
    
#     # 计算powerW
#     powerW = 0.0
#     for j in range(nfft):
#         powerW += temp_real[j] * temp_real[j]
#     cdef double inv_powerW = 1.0 / powerW
    
#     # 处理两个分量（径向和切向）
#     for index in range(2):
#         has_solution = False
#         current_bic = 0.0
#         min_bic = 1e20
#         for j in range(nfft):
#             temp_rf[j] = 0.0
#         # 确定通道索引
#         chan = 1 if (odina_flag == 2 and index == 0) else (2 if (odina_flag == 2) else index)
        
#         # 初始化U
#         for j in range(nfft):
#             U[j] = stream_ptr[chan*nfft + j]
        
#         # FFT运算
#         rfft_fftw_reuse(plans, work, U, temp_complex)
        
#         # 频域滤波
#         for j in range(nfft):
#             temp_re = temp_complex[j].real * gaussF_ptr[j].real - temp_complex[j].imag * gaussF_ptr[j].imag
#             temp_im = temp_complex[j].real * gaussF_ptr[j].imag + temp_complex[j].imag * gaussF_ptr[j].real
#             temp_complex[j].real = temp_re
#             temp_complex[j].imag = temp_im
        
#         # IFFT运算
#         irfft_fftw_reuse(plans, work, temp_complex, U)
        
#         # 计算powerU
#         powerU = 0.0
#         for j in range(nfft):
#             powerU += U[j] * U[j]
        
#         # 初始化P0和temp_real (作为残差R)
#         for j in range(nfft):
#             P0[j] = 0.0
#             temp_real[j] = U[j]
        
#         sumsq_i = 1.0
#         d_error = 100.0 * powerU + minderr
        
#         # 开始迭代反卷积
#         for it in range(itmax):
#             if it == 0 and fabs(d_error) == minderr:
#                 # 初始解处理
#                 for j in range(nfft):
#                     temp_rf[j] = P0[j]
#                 min_bic = log(nfft) * 2 + nfft * log(powerU / nfft)
#                 has_solution = True
#                 break
#             elif fabs(d_error) > minderr:
#                 # FFT卷积运算
#                 rfft_fftw_reuse(plans, work, temp_real, temp_complex)
                
#                 # 复数乘法 (A * conj(B))
#                 for j in range(nfft):
#                     temp_re = (temp_complex[j].real * Wf[j].real + temp_complex[j].imag * Wf[j].imag)
#                     temp_im = (temp_complex[j].imag * Wf[j].real - temp_complex[j].real * Wf[j].imag)
#                     temp_complex[j].real = temp_re * inv_powerW
#                     temp_complex[j].imag = temp_im * inv_powerW
                
#                 # IFFT运算
#                 irfft_fftw_reuse(plans, work, temp_complex, temp_real)
                
#                 # 寻找最大值
#                 index_k = 0
#                 max_val = 0.0
#                 for j in range(half):
#                     if fabs(temp_real[j]) > max_val:
#                         max_val = fabs(temp_real[j])
#                         index_k = j
                
#                 amp = temp_real[index_k]
#                 P0[index_k] += amp
                
#                 # 计算预测值
#                 rfft_fftw_reuse(plans, work, P0, temp_complex)
                
#                 # 频域滤波
#                 for j in range(nfft):
#                     creal = temp_complex[j].real
#                     cimag = temp_complex[j].imag
#                     # 复数乘法: (gaussF * W0f * dt_bare^2)
#                     filter_real = gaussF_ptr[j].real * W0f[j].real - gaussF_ptr[j].imag * W0f[j].imag
#                     filter_imag = gaussF_ptr[j].real * W0f[j].imag + gaussF_ptr[j].imag * W0f[j].real
#                     temp_complex[j].real = (creal * filter_real - cimag * filter_imag)
#                     temp_complex[j].imag = (creal * filter_imag + cimag * filter_real)
                
#                 # IFFT运算
#                 irfft_fftw_reuse(plans, work, temp_complex, P)
                
#                 # 计算残差
#                 powerR = 0.0
#                 non_zero = 0
#                 for j in range(nfft):
#                     diff = U[j] - P[j]
#                     temp_real[j] = diff
#                     powerR += diff * diff
#                     if P0[j] != 0.0:
#                         non_zero += 1
                        
#                 sumsq = powerR / powerU
                
#                 # 计算当前BIC
#                 current_bic = log(nfft) * non_zero * 2 + nfft * log(powerR / nfft)
#                 # 更新最优解
#                 if current_bic < min_bic:
#                     min_bic = current_bic
#                     for j in range(nfft):
#                         temp_rf[j] = P0[j]
#                     has_solution = True

#                 # 更新误差
#                 d_error = 100.0 * (sumsq_i - sumsq)
#                 sumsq_i = sumsq
#             else:
#                 break
        
#         # 复制最优解到输出
#         if has_solution:
#             for j in range(nfft):
#                 rf[j] = temp_rf[j]
#         else:
#             for j in range(nfft):
#                 rf[j] = 0.0
        
#         # FFT运算
#         rfft_fftw_reuse(plans, work, rf, temp_complex)
        
#         # 频域滤波
#         for j in range(nfft):
#             temp_re = temp_complex[j].real * gaussF_nor_ptr[j].real - temp_complex[j].imag * gaussF_nor_ptr[j].imag
#             temp_im = temp_complex[j].real * gaussF_nor_ptr[j].imag + temp_complex[j].imag * gaussF_nor_ptr[j].real
#             temp_complex[j].real = temp_re
#             temp_complex[j].imag = temp_im
        
#         # IFFT运算
#         irfft_fftw_reuse(plans, work, temp_complex, rf)
        
#         # 相位校正
#         shift_i = <int>(round(tshift / dt_bare))
#         for j in range(nfft):
#             phase_shift[j].real = cos(2 * M_PI * j * shift_i / nfft)
#             phase_shift[j].imag = -sin(2 * M_PI * j * shift_i / nfft)
        
#         # FFT运算
#         rfft_fftw_reuse(plans, work, rf, temp_complex)
        
#         # 相位校正
#         for j in range(nfft):
#             temp_re = temp_complex[j].real * phase_shift[j].real - temp_complex[j].imag * phase_shift[j].imag
#             temp_im = temp_complex[j].real * phase_shift[j].imag + temp_complex[j].imag * phase_shift[j].real
#             temp_complex[j].real = temp_re
#             temp_complex[j].imag = temp_im
        
#         # IFFT运算
#         irfft_fftw_reuse(plans, work, temp_complex, rf)
        
#         # 归一化
#         norm = 1.0 / (cos(2 * M_PI * shift_i / nfft) + 1e-10)  # 避免除以零
#         for j in range(nfft):
#             rf[j] *= norm
        
#         # 存储结果
#         for j in range(nfft):
#             output_ptr[index*nfft + j] = rf[j]