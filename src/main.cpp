// SciBaseCPP — Demo: Scientific Computing in Modern C++
//
// This file demonstrates the use of several libraries pulled in via
// CMake FetchContent, showing that expressive scientific C++ is possible
// without any manual dependency installation.

#include <experimental/mdspan>
#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <nlopt.hpp>
#include <pocketfft_hdronly.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sundials/sundials_core.hpp>

#include <vector>
#include <complex>
#include <cmath>
#include <numbers>
#include <numeric>
#include <functional>

// Kokkos reference implementation places mdspan, extents, dextents etc. in std::
// when included via <experimental/mdspan>. The std::experimental namespace only
// has a few backward-compatibility aliases (mdspan, extents, layout_*).
// So we use std:: directly — this also means the code is forward-compatible with
// compilers that ship native C++23 <mdspan>.

// ============================================================================
// 1. mdspan: Multidimensional array views (C++23)
// ============================================================================
void demo_mdspan() {
    fmt::println("=== mdspan: Multidimensional Array Views ===\n");

    // Create a 4x3 matrix backed by a flat vector
    std::vector<double> data(12);
    auto mat = std::mdspan<double, std::dextents<size_t, 2>>(data.data(), 4, 3);

    // Fill with values: mat[i,j] = sin(i) * cos(j)
    for (size_t i = 0; i < mat.extent(0); i++)
        for (size_t j = 0; j < mat.extent(1); j++)
            mat[i, j] = std::sin(static_cast<double>(i))
                       * std::cos(static_cast<double>(j));

    // Print the matrix
    fmt::println("  4x3 matrix of sin(i)*cos(j):");
    for (size_t i = 0; i < mat.extent(0); i++) {
        fmt::print("    ");
        for (size_t j = 0; j < mat.extent(1); j++)
            fmt::print("{:8.4f} ", mat[i, j]);
        fmt::println("");
    }

    // mdspan with compile-time extents (3D, first dim fixed at 2)
    std::vector<double> vol(2 * 3 * 4);
    auto tensor = std::mdspan<double,
        std::extents<size_t, 2, std::dynamic_extent, std::dynamic_extent>>(
            vol.data(), 3, 4);

    fmt::println("  3D tensor rank: {}, extents: {}x{}x{}",
        tensor.rank(), tensor.extent(0), tensor.extent(1), tensor.extent(2));
    fmt::println("");
}

// ============================================================================
// 2. Eigen: Linear Algebra
// ============================================================================
void demo_eigen() {
    fmt::println("=== Eigen: Linear Algebra ===\n");

    // Solve a 3x3 linear system Ax = b
    Eigen::Matrix3d A;
    A << 2, -1,  0,
        -1,  2, -1,
         0, -1,  2;

    Eigen::Vector3d b(1.0, 0.0, 1.0);
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    fmt::println("  Solving Ax = b where A is a tridiagonal matrix:");
    fmt::println("  x = [{:.4f}, {:.4f}, {:.4f}]", x(0), x(1), x(2));
    fmt::println("  Residual |Ax - b| = {:.2e}", (A * x - b).norm());

    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(A);
    auto eigenvalues = solver.eigenvalues();
    fmt::println("  Eigenvalues: [{:.4f}, {:.4f}, {:.4f}]",
        eigenvalues(0), eigenvalues(1), eigenvalues(2));

    // SVD of a rectangular matrix
    Eigen::MatrixXd B(4, 2);
    B << 1, 2,
         3, 4,
         5, 6,
         7, 8;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    fmt::println("  SVD singular values of 4x2 matrix: [{:.4f}, {:.4f}]",
        svd.singularValues()(0), svd.singularValues()(1));
    fmt::println("");
}

// ============================================================================
// 3. Eigen + mdspan interop: zero-copy views
// ============================================================================
void demo_eigen_mdspan_interop() {
    fmt::println("=== Eigen + mdspan Interop ===\n");

    // Eigen manages the storage, mdspan provides a multidimensional view
    Eigen::VectorXd vec = Eigen::VectorXd::LinSpaced(12, 0, 11);

    // Wrap Eigen's raw data with an mdspan 3x4 view
    auto mat_view = std::mdspan<double, std::dextents<size_t, 2>>(
        vec.data(), 3, 4);

    fmt::println("  Eigen vector viewed as 3x4 mdspan:");
    for (size_t i = 0; i < mat_view.extent(0); i++) {
        fmt::print("    ");
        for (size_t j = 0; j < mat_view.extent(1); j++)
            fmt::print("{:5.1f} ", mat_view[i, j]);
        fmt::println("");
    }
    fmt::println("");
}

// ============================================================================
// 4. NLopt: Nonlinear Optimization
// ============================================================================
// Minimize the Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
double rosenbrock(const std::vector<double>& x, std::vector<double>& grad, void*) {
    double a = 1.0 - x[0];
    double b = x[1] - x[0] * x[0];
    if (!grad.empty()) {
        grad[0] = -2.0 * a - 400.0 * x[0] * b;
        grad[1] = 200.0 * b;
    }
    return a * a + 100.0 * b * b;
}

void demo_nlopt() {
    fmt::println("=== NLopt: Nonlinear Optimization ===\n");

    nlopt::opt opt(nlopt::LD_LBFGS, 2);
    opt.set_min_objective(rosenbrock, nullptr);
    opt.set_xtol_rel(1e-12);

    std::vector<double> x = {-1.0, -1.0};
    double minf;
    opt.optimize(x, minf);

    fmt::println("  Rosenbrock minimum: f({:.6f}, {:.6f}) = {:.2e}",
        x[0], x[1], minf);
    fmt::println("  (exact: f(1, 1) = 0)");
    fmt::println("");
}

// ============================================================================
// 5. PocketFFT: Fast Fourier Transforms (BSD-licensed, same engine as NumPy)
// ============================================================================
void demo_pocketfft() {
    fmt::println("=== PocketFFT: Fast Fourier Transform ===\n");

    // Create a signal: sum of two sinusoids at 5 Hz and 12 Hz
    const size_t N = 64;
    const double dt = 1.0 / 64.0;  // 64 samples/sec
    std::vector<std::complex<double>> signal(N);
    for (size_t i = 0; i < N; i++) {
        double t = i * dt;
        signal[i] = 3.0 * std::sin(2.0 * std::numbers::pi * 5.0 * t)
                  + 1.5 * std::sin(2.0 * std::numbers::pi * 12.0 * t);
    }

    // Perform forward FFT using PocketFFT
    pocketfft::shape_t shape = {N};
    pocketfft::stride_t stride_in = {static_cast<ptrdiff_t>(sizeof(std::complex<double>))};
    pocketfft::stride_t stride_out = stride_in;

    std::vector<std::complex<double>> result(N);
    pocketfft::c2c(shape, stride_in, stride_out, {0}, pocketfft::FORWARD,
                   signal.data(), result.data(), 1.0);

    // Print the dominant frequencies
    fmt::println("  Signal: 3·sin(2π·5t) + 1.5·sin(2π·12t), N={}", N);
    fmt::println("  Top FFT magnitudes (freq, |X[k]|/N):");
    for (size_t k = 0; k < N / 2; k++) {
        double mag = std::abs(result[k]) / N;
        if (mag > 0.1)
            fmt::println("    f = {:2} Hz : magnitude = {:.4f}", k, mag);
    }
    fmt::println("");
}

// ============================================================================
// 6. SUNDIALS/CVODE: Stiff & nonstiff ODE integration
// ============================================================================
// Right-hand side: dy/dt = -y (exponential decay)
static int cvode_rhs(sunrealtype /*t*/, N_Vector y, N_Vector ydot, void* /*user_data*/) {
    NV_Ith_S(ydot, 0) = -NV_Ith_S(y, 0);
    return 0;
}

void demo_cvode() {
    fmt::println("=== SUNDIALS/CVODE: ODE Integration ===\n");

    // Create SUNDIALS context
    sundials::Context sunctx;

    // Initial condition: y(0) = 1
    N_Vector y = N_VNew_Serial(1, sunctx);
    NV_Ith_S(y, 0) = 1.0;

    // Create CVODE solver (Adams method for nonstiff, BDF for stiff)
    void* cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
    CVodeInit(cvode_mem, cvode_rhs, 0.0, y);
    CVodeSStolerances(cvode_mem, 1e-12, 1e-14);

    // Create dense linear solver (needed even for Adams when using functional iteration)
    SUNMatrix A = SUNDenseMatrix(1, 1, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    CVodeSetLinearSolver(cvode_mem, LS, A);

    // Integrate to t = 5
    sunrealtype t_final;
    CVode(cvode_mem, 5.0, y, &t_final, CV_NORMAL);

    double y_final = NV_Ith_S(y, 0);
    double exact = std::exp(-5.0);
    fmt::println("  dy/dt = -y, y(0) = 1  (Adams-Moulton)");
    fmt::println("  y(5) = {:.12f}  (exact: {:.12f})", y_final, exact);
    fmt::println("  Error: {:.2e}", std::abs(y_final - exact));

    // Cleanup
    N_VDestroy(y);
    CVodeFree(&cvode_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    fmt::println("");
}

// ============================================================================
// 7. A simple ODE integrator (RK4) — no external dependency needed
// ============================================================================
// Solve dy/dt = f(t, y) using classical Runge-Kutta
template <typename F>
std::vector<double> rk4(F f, double y0, double t0, double t1, int n) {
    double h = (t1 - t0) / n;
    std::vector<double> result;
    result.reserve(n + 1);
    double y = y0, t = t0;
    result.push_back(y);
    for (int i = 0; i < n; i++) {
        double k1 = h * f(t, y);
        double k2 = h * f(t + h / 2, y + k1 / 2);
        double k3 = h * f(t + h / 2, y + k2 / 2);
        double k4 = h * f(t + h, y + k3);
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        t += h;
        result.push_back(y);
    }
    return result;
}

void demo_ode() {
    fmt::println("=== ODE Integration (RK4) ===\n");

    // Solve dy/dt = -y, y(0) = 1  =>  y(t) = exp(-t)
    auto f = [](double /*t*/, double y) { return -y; };
    auto sol = rk4(f, 1.0, 0.0, 5.0, 1000);

    double y_final = sol.back();
    double exact = std::exp(-5.0);
    fmt::println("  dy/dt = -y, y(0) = 1");
    fmt::println("  y(5) = {:.10f}  (exact: {:.10f})", y_final, exact);
    fmt::println("  Error: {:.2e}", std::abs(y_final - exact));
    fmt::println("");
}

// ============================================================================
// 8. Special functions (C++17 standard library)
// ============================================================================
void demo_special_functions() {
    fmt::println("=== C++17 Special Functions ===\n");

    double x = 0.5;
    fmt::println("  x = {}", x);
    fmt::println("  tgamma(x)  = {:.10f}  (Gamma function)", std::tgamma(x));
    fmt::println("  lgamma(x)  = {:.10f}  (log Gamma)", std::lgamma(x));
    fmt::println("  erf(x)     = {:.10f}  (error function)", std::erf(x));
    fmt::println("  erfc(x)    = {:.10f}  (complementary error function)", std::erfc(x));

    // Bessel functions (C++17)
    fmt::println("  cyl_bessel_j(0, x) = {:.10f}  (J0 Bessel)", std::cyl_bessel_j(0, x));
    fmt::println("  cyl_bessel_j(1, x) = {:.10f}  (J1 Bessel)", std::cyl_bessel_j(1, x));

    // Legendre polynomials
    fmt::println("  legendre(2, x)     = {:.10f}  (P2 Legendre)", std::legendre(2, x));
    fmt::println("  legendre(3, x)     = {:.10f}  (P3 Legendre)", std::legendre(3, x));
    fmt::println("");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    fmt::println("╔══════════════════════════════════════════════════╗");
    fmt::println("║  SciBaseCPP — Scientific Computing in C++23    ║");
    fmt::println("╚══════════════════════════════════════════════════╝\n");

    demo_mdspan();
    demo_eigen();
    demo_eigen_mdspan_interop();
    demo_nlopt();
    demo_pocketfft();
    demo_cvode();
    demo_ode();
    demo_special_functions();

    fmt::println("All demos completed successfully!");
    return 0;
}
