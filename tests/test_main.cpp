// SciBaseCPP — Test suite using Catch2

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <experimental/mdspan>
#include <Eigen/Dense>
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

using Catch::Approx;

// ============================================================================
// mdspan tests
// ============================================================================

TEST_CASE("mdspan: basic 2D access", "[mdspan]") {
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    auto mat = std::mdspan<double, std::dextents<size_t, 2>>(data.data(), 2, 3);

    REQUIRE(mat.rank() == 2);
    REQUIRE(mat.extent(0) == 2);
    REQUIRE(mat.extent(1) == 3);
    REQUIRE(mat[0, 0] == 1.0);
    REQUIRE(mat[0, 2] == 3.0);
    REQUIRE(mat[1, 0] == 4.0);
    REQUIRE(mat[1, 2] == 6.0);
}

TEST_CASE("mdspan: compile-time extents", "[mdspan]") {
    std::array<double, 6> data = {10, 20, 30, 40, 50, 60};
    auto mat = std::mdspan<double, std::extents<size_t, 2, 3>>(data.data());

    REQUIRE(mat.rank() == 2);
    REQUIRE(mat.extent(0) == 2);
    REQUIRE(mat.extent(1) == 3);
    REQUIRE(mat[1, 1] == 50.0);
}

TEST_CASE("mdspan: 3D tensor", "[mdspan]") {
    std::vector<double> data(24);
    auto t = std::mdspan<double, std::dextents<size_t, 3>>(data.data(), 2, 3, 4);

    REQUIRE(t.rank() == 3);
    REQUIRE(t.size() == 24);

    t[1, 2, 3] = 42.0;
    REQUIRE(t[1, 2, 3] == 42.0);
}

// ============================================================================
// Eigen tests
// ============================================================================

TEST_CASE("Eigen: solve linear system", "[eigen]") {
    Eigen::Matrix3d A;
    A << 2, -1,  0,
        -1,  2, -1,
         0, -1,  2;
    Eigen::Vector3d b(1, 0, 1);

    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
    double residual = (A * x - b).norm();

    REQUIRE(residual == Approx(0.0).margin(1e-14));
}

TEST_CASE("Eigen: eigenvalue decomposition", "[eigen]") {
    // Symmetric 2x2 matrix with known eigenvalues
    Eigen::Matrix2d A;
    A << 3, 1,
         1, 3;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(A);
    auto evals = solver.eigenvalues();

    REQUIRE(evals(0) == Approx(2.0));
    REQUIRE(evals(1) == Approx(4.0));
}

TEST_CASE("Eigen: SVD", "[eigen]") {
    Eigen::MatrixXd A(3, 2);
    A << 1, 0,
         0, 1,
         0, 0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    auto s = svd.singularValues();

    REQUIRE(s(0) == Approx(1.0));
    REQUIRE(s(1) == Approx(1.0));
}

// ============================================================================
// Eigen + mdspan interop
// ============================================================================

TEST_CASE("Eigen + mdspan: zero-copy interop", "[interop]") {
    Eigen::VectorXd vec(6);
    vec << 1, 2, 3, 4, 5, 6;

    auto mat = std::mdspan<double, std::dextents<size_t, 2>>(vec.data(), 2, 3);

    // mdspan sees the same data as Eigen
    REQUIRE(mat[0, 0] == vec(0));
    REQUIRE(mat[1, 2] == vec(5));

    // Modifying through mdspan should be visible in Eigen
    mat[0, 1] = 99.0;
    REQUIRE(vec(1) == 99.0);
}

// ============================================================================
// NLopt tests
// ============================================================================

TEST_CASE("NLopt: minimize Rosenbrock", "[nlopt]") {
    auto rosenbrock = [](const std::vector<double>& x,
                         std::vector<double>& grad, void*) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        if (!grad.empty()) {
            grad[0] = -2.0 * a - 400.0 * x[0] * b;
            grad[1] = 200.0 * b;
        }
        return a * a + 100.0 * b * b;
    };

    nlopt::opt opt(nlopt::LD_LBFGS, 2);
    opt.set_min_objective(rosenbrock, nullptr);
    opt.set_xtol_rel(1e-12);

    std::vector<double> x = {-1.0, -1.0};
    double minf;
    opt.optimize(x, minf);

    REQUIRE(x[0] == Approx(1.0).margin(1e-6));
    REQUIRE(x[1] == Approx(1.0).margin(1e-6));
    REQUIRE(minf == Approx(0.0).margin(1e-10));
}

// ============================================================================
// PocketFFT tests
// ============================================================================

TEST_CASE("PocketFFT: forward and inverse c2c round-trip", "[fft]") {
    const size_t N = 128;
    std::vector<std::complex<double>> input(N);
    for (size_t i = 0; i < N; i++)
        input[i] = std::sin(2.0 * std::numbers::pi * 5.0 * i / N);

    pocketfft::shape_t shape = {N};
    pocketfft::stride_t stride = {static_cast<ptrdiff_t>(sizeof(std::complex<double>))};

    // Forward
    std::vector<std::complex<double>> freq(N);
    pocketfft::c2c(shape, stride, stride, {0}, pocketfft::FORWARD,
                   input.data(), freq.data(), 1.0);

    // Inverse (with 1/N normalization)
    std::vector<std::complex<double>> recovered(N);
    pocketfft::c2c(shape, stride, stride, {0}, pocketfft::BACKWARD,
                   freq.data(), recovered.data(), 1.0 / N);

    // Round-trip should recover original signal
    for (size_t i = 0; i < N; i++)
        REQUIRE(std::abs(recovered[i] - input[i]) < 1e-12);
}

TEST_CASE("PocketFFT: detects sinusoid frequency", "[fft]") {
    const size_t N = 64;
    std::vector<std::complex<double>> signal(N);
    // Pure 8 Hz tone
    for (size_t i = 0; i < N; i++)
        signal[i] = std::sin(2.0 * std::numbers::pi * 8.0 * i / N);

    pocketfft::shape_t shape = {N};
    pocketfft::stride_t stride = {static_cast<ptrdiff_t>(sizeof(std::complex<double>))};
    std::vector<std::complex<double>> result(N);
    pocketfft::c2c(shape, stride, stride, {0}, pocketfft::FORWARD,
                   signal.data(), result.data(), 1.0);

    // Peak should be at bin 8
    double max_mag = 0;
    size_t max_bin = 0;
    for (size_t k = 1; k < N / 2; k++) {
        double mag = std::abs(result[k]);
        if (mag > max_mag) { max_mag = mag; max_bin = k; }
    }
    REQUIRE(max_bin == 8);
}

// ============================================================================
// SUNDIALS/CVODE tests
// ============================================================================

static int test_cvode_rhs(sunrealtype /*t*/, N_Vector y, N_Vector ydot, void*) {
    NV_Ith_S(ydot, 0) = -NV_Ith_S(y, 0);
    return 0;
}

TEST_CASE("CVODE: exponential decay", "[cvode]") {
    sundials::Context sunctx;

    N_Vector y = N_VNew_Serial(1, sunctx);
    NV_Ith_S(y, 0) = 1.0;

    void* cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
    CVodeInit(cvode_mem, test_cvode_rhs, 0.0, y);
    CVodeSStolerances(cvode_mem, 1e-12, 1e-14);

    SUNMatrix A = SUNDenseMatrix(1, 1, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    CVodeSetLinearSolver(cvode_mem, LS, A);

    sunrealtype t_final;
    CVode(cvode_mem, 5.0, y, &t_final, CV_NORMAL);

    double y_final = NV_Ith_S(y, 0);
    REQUIRE(y_final == Approx(std::exp(-5.0)).margin(1e-10));

    N_VDestroy(y);
    CVodeFree(&cvode_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
}

// ============================================================================
// C++17 special functions
// ============================================================================

TEST_CASE("Special functions: Gamma", "[special]") {
    REQUIRE(std::tgamma(1.0) == Approx(1.0));
    REQUIRE(std::tgamma(0.5) == Approx(std::sqrt(std::numbers::pi)));
    REQUIRE(std::tgamma(5.0) == Approx(24.0));
}

TEST_CASE("Special functions: Bessel J0", "[special]") {
    // J0(0) = 1
    REQUIRE(std::cyl_bessel_j(0, 0.0) == Approx(1.0));
    // First zero of J0 ≈ 2.4048
    REQUIRE(std::abs(std::cyl_bessel_j(0, 2.4048)) < 1e-4);
}

TEST_CASE("Special functions: Legendre", "[special]") {
    // P0(x) = 1, P1(x) = x, P2(x) = (3x^2 - 1)/2
    REQUIRE(std::legendre(0, 0.5) == Approx(1.0));
    REQUIRE(std::legendre(1, 0.5) == Approx(0.5));
    REQUIRE(std::legendre(2, 0.5) == Approx(-0.125));
}
