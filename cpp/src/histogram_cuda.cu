// CUDA histogram kernel. Compiled only with TIDEGBM_CUDA (see setup.py / scripts/build_gpu_wheel.sh).
#ifdef TIDEGBM_CUDA
#include "tidegbm/histogram.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

namespace tidegbm {

__device__ inline void atomic_add_double(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    unsigned long long int* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

__global__ void hist_atomic_kernel(const uint16_t* xb, const int* idx, int n_idx, int n_features,
                                   int n_bins, const double* g, const double* h, double* G, double* H,
                                   long long* C) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_idx) return;
    const int i = idx[k];
    const uint16_t* row = xb + static_cast<size_t>(i) * n_features;
    const double gi = g[i];
    const double hi = h[i];
    for (int f = 0; f < n_features; ++f) {
        const int b = static_cast<int>(row[f]);
        const size_t slot = static_cast<size_t>(f) * n_bins + b;
        atomic_add_double(G + slot, gi);
        atomic_add_double(H + slot, hi);
        atomicAdd(reinterpret_cast<unsigned long long*>(C + slot), 1ull);
    }
}

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw TideError(std::string(what) + ": " + cudaGetErrorString(e));
}

struct GpuHistState {
    const u16* host_xb = nullptr;
    const f64* host_g = nullptr;
    const f64* host_h = nullptr;
    int n_rows = 0;
    int n_features = 0;
    u16* d_xb = nullptr;
    f64 *d_g = nullptr, *d_h = nullptr;
    i32* d_idx = nullptr;
    int idx_cap = 0;
    f64 *d_G = nullptr, *d_H = nullptr;
    long long* d_C = nullptr;
    size_t hist_cap = 0;

    void release() {
        if (d_xb) cudaFree(d_xb);
        if (d_g) cudaFree(d_g);
        if (d_h) cudaFree(d_h);
        if (d_idx) cudaFree(d_idx);
        if (d_G) cudaFree(d_G);
        if (d_H) cudaFree(d_H);
        if (d_C) cudaFree(d_C);
        *this = GpuHistState();
    }
};

static GpuHistState& gpu_state() {
    static GpuHistState st;
    return st;
}

void build_hist_gpu(const u16* xb, int n_features, const i32* idx, int n_idx, const f64* g,
                    const f64* h, Hist& out, int n_rows) {
    out.zero();
    if (n_idx <= 0) return;
    if (n_rows <= 0) {
        // packed fallback: treat idx as dense 0..n_idx-1 over a packed xb
        n_rows = n_idx;
    }
    const size_t hist_sz = static_cast<size_t>(n_features) * out.n_bins;
    GpuHistState& st = gpu_state();

    if (st.host_xb != xb || st.n_rows != n_rows || st.n_features != n_features) {
        if (st.d_xb) cudaFree(st.d_xb);
        st.d_xb = nullptr;
        const size_t xb_bytes = static_cast<size_t>(n_rows) * n_features * sizeof(u16);
        check_cuda(cudaMalloc(&st.d_xb, xb_bytes), "cudaMalloc xb");
        check_cuda(cudaMemcpy(st.d_xb, xb, xb_bytes, cudaMemcpyHostToDevice), "H2D xb");
        st.host_xb = xb;
        st.n_rows = n_rows;
        st.n_features = n_features;
        st.host_g = nullptr;
        st.host_h = nullptr;
    }
    if (st.host_g != g || st.d_g == nullptr) {
        if (st.d_g) cudaFree(st.d_g);
        st.d_g = nullptr;
        check_cuda(cudaMalloc(&st.d_g, static_cast<size_t>(n_rows) * sizeof(f64)), "cudaMalloc g");
        st.host_g = g;
    }
    check_cuda(cudaMemcpy(st.d_g, g, static_cast<size_t>(n_rows) * sizeof(f64), cudaMemcpyHostToDevice),
               "H2D g");
    if (st.host_h != h || st.d_h == nullptr) {
        if (st.d_h) cudaFree(st.d_h);
        st.d_h = nullptr;
        check_cuda(cudaMalloc(&st.d_h, static_cast<size_t>(n_rows) * sizeof(f64)), "cudaMalloc h");
        st.host_h = h;
    }
    check_cuda(cudaMemcpy(st.d_h, h, static_cast<size_t>(n_rows) * sizeof(f64), cudaMemcpyHostToDevice),
               "H2D h");

    if (st.idx_cap < n_idx) {
        if (st.d_idx) cudaFree(st.d_idx);
        st.d_idx = nullptr;
        st.idx_cap = std::max(n_idx, st.idx_cap * 2);
        check_cuda(cudaMalloc(&st.d_idx, static_cast<size_t>(st.idx_cap) * sizeof(i32)), "cudaMalloc idx");
    }
    check_cuda(cudaMemcpy(st.d_idx, idx, static_cast<size_t>(n_idx) * sizeof(i32), cudaMemcpyHostToDevice),
               "H2D idx");

    if (st.hist_cap < hist_sz) {
        if (st.d_G) cudaFree(st.d_G);
        if (st.d_H) cudaFree(st.d_H);
        if (st.d_C) cudaFree(st.d_C);
        st.d_G = st.d_H = nullptr;
        st.d_C = nullptr;
        st.hist_cap = hist_sz;
        check_cuda(cudaMalloc(&st.d_G, hist_sz * sizeof(f64)), "cudaMalloc G");
        check_cuda(cudaMalloc(&st.d_H, hist_sz * sizeof(f64)), "cudaMalloc H");
        check_cuda(cudaMalloc(&st.d_C, hist_sz * sizeof(long long)), "cudaMalloc C");
    }
    check_cuda(cudaMemset(st.d_G, 0, hist_sz * sizeof(f64)), "memset G");
    check_cuda(cudaMemset(st.d_H, 0, hist_sz * sizeof(f64)), "memset H");
    check_cuda(cudaMemset(st.d_C, 0, hist_sz * sizeof(long long)), "memset C");

    const int block = 256;
    const int grid = std::max(1, (n_idx + block - 1) / block);
    hist_atomic_kernel<<<grid, block>>>(st.d_xb, st.d_idx, n_idx, n_features, out.n_bins, st.d_g, st.d_h,
                                        st.d_G, st.d_H, st.d_C);
    check_cuda(cudaGetLastError(), "hist kernel launch");
    check_cuda(cudaDeviceSynchronize(), "hist kernel");
    // Device keeps three flat planes; the host histogram is interleaved, so fold on copy-back.
    static thread_local std::vector<f64> hG, hH;
    static thread_local std::vector<long long> hC;
    hG.resize(hist_sz);
    hH.resize(hist_sz);
    hC.resize(hist_sz);
    check_cuda(cudaMemcpy(hG.data(), st.d_G, hist_sz * sizeof(f64), cudaMemcpyDeviceToHost), "D2H G");
    check_cuda(cudaMemcpy(hH.data(), st.d_H, hist_sz * sizeof(f64), cudaMemcpyDeviceToHost), "D2H H");
    check_cuda(cudaMemcpy(hC.data(), st.d_C, hist_sz * sizeof(long long), cudaMemcpyDeviceToHost),
               "D2H C");
    HBin* B = out.B.data();
    for (size_t i = 0; i < hist_sz; ++i) {
        B[i].g = hG[i];
        B[i].h = hH[i];
        B[i].c = static_cast<i64>(hC[i]);
    }
}

}  // namespace tidegbm
#endif
