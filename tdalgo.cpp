#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <ctime>

struct TDResult {
    std::vector<double> x1;
    std::vector<double> x2;
};

TDResult simple_linear_td2(const std::vector<double>& v_in, double h, double R, double x1_0 = 0.0, double x2_0 = 0.0) {
    if (v_in.empty())
        return {{}, {}};

    size_t n = v_in.size();
    TDResult result;
    result.x1.resize(n);
    result.x2.resize(n);

    double current_x1 = x1_0;
    double current_x2 = x2_0;

    double alpha0 = 2.0 * std::sqrt(R);
    double alpha1 = R;

    for (size_t k = 0; k < n; ++k) {
        double error = current_x1 - v_in[k];
        double u = -alpha0 * error - alpha1 * (current_x2 / R);
        current_x1 = current_x1 + h * current_x2;
        current_x2 = current_x2 + h * u;
        result.x1[k] = current_x1;
        result.x2[k] = current_x2;
    }
    return result;
}

std::vector<double> generate_test_signal(size_t num_points, double h, double amplitude, double frequency, double noise_level) {
    std::vector<double> signal(num_points);
    const double PI = std::acos(-1.0);

    for (size_t i = 0; i < num_points; ++i) {
        signal[i] = amplitude * std::sin(2.0 * PI * frequency * (double)i * h);
        if (noise_level > 0)
            signal[i] += noise_level * (((double)rand() / RAND_MAX) * 2.0 - 1.0);
    }
    return signal;
}

void print_signal(const std::string& name, const std::vector<double>& signal, int precision = 4, size_t points_to_show = 10) {
    std::cout << name << ": [";
    if (signal.empty()) {
        std::cout << " (empty)";
    } else {
        size_t limit = std::min(signal.size(), points_to_show);
        for (size_t i = 0; i < limit; ++i) {
            std::cout << std::fixed << std::setprecision(precision) << signal[i];
            if (i < limit - 1)
                std::cout << ", ";
        }
        if (signal.size() > limit)
            std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

int main() {
    srand(time(NULL));

    const size_t num_signals = 4;
    const size_t signal_length = 100;
    const double h_step = 0.01;
    const double R_param = 1000.0;
    const double amplitude = 1.0;
    const double frequency = 1.0;
    const double noise = 0.1;

    std::vector<std::vector<double>> input_signals(num_signals);
    for (size_t i = 0; i < num_signals; ++i)
        input_signals[i] = generate_test_signal(signal_length, h_step, amplitude + (double)i * 0.1, frequency + (double)i * 0.05, noise);

    std::vector<TDResult> results_sequential(num_signals);
    std::vector<TDResult> results_parallel(num_signals);

    std::cout << "--- Sequential Processing ---" << std::endl;
    double start_time_seq = omp_get_wtime();
    for (size_t i = 0; i < num_signals; ++i) {
        if (!input_signals[i].empty())
            results_sequential[i] = simple_linear_td2(input_signals[i], h_step, R_param, input_signals[i][0]);
    }
    double end_time_seq = omp_get_wtime();
    std::cout << "Sequential time: " << std::fixed << std::setprecision(6) << (end_time_seq - start_time_seq) * 1000.0 << " ms" << std::endl;

    for (size_t i = 0; i < num_signals; ++i) {
        std::cout << "\nSignal " << i << ":" << std::endl;
        print_signal("  Input      ", input_signals[i]);
        print_signal("  TD x1 (seq)", results_sequential[i].x1);
        print_signal("  TD x2 (seq)", results_sequential[i].x2);
    }
    std::cout << std::endl;

    std::cout << "--- Parallel Processing (OpenMP) ---" << std::endl;
    int num_threads_to_use = omp_get_max_threads();
    std::cout << "Using " << num_threads_to_use << " OpenMP threads." << std::endl;

    double start_time_par = omp_get_wtime();
    #pragma omp parallel for num_threads(num_threads_to_use) schedule(static)
    for (size_t i = 0; i < num_signals; ++i) {
        if (!input_signals[i].empty())
            results_parallel[i] = simple_linear_td2(input_signals[i], h_step, R_param, input_signals[i][0]);
    }
    double end_time_par = omp_get_wtime();
    std::cout << "Parallel time: " << std::fixed << std::setprecision(6) << (end_time_par - start_time_par) * 1000.0 << " ms" << std::endl;

    for (size_t i = 0; i < num_signals; ++i) {
        std::cout << "\nSignal " << i << ":" << std::endl;
        print_signal("  TD x1 (par)", results_parallel[i].x1);
        print_signal("  TD x2 (par)", results_parallel[i].x2);
    }
    std::cout << std::endl;

    bool all_match = true;
    const double tolerance = 1e-9;
    if (results_sequential.size() == results_parallel.size()) {
        for (size_t i = 0; i < num_signals; ++i) {
            if (results_sequential[i].x1.size() != results_parallel[i].x1.size() ||
                results_sequential[i].x2.size() != results_parallel[i].x2.size()) {
                all_match = false;
                std::cerr << "Size mismatch for results of signal " << i << std::endl;
                break;
            }
            for (size_t j = 0; j < results_sequential[i].x1.size(); ++j) {
                if (std::abs(results_sequential[i].x1[j] - results_parallel[i].x1[j]) > tolerance ||
                    std::abs(results_sequential[i].x2[j] - results_parallel[i].x2[j]) > tolerance) {
                    all_match = false;
                    std::cerr << "Value mismatch for signal " << i << " at point " << j << std::endl;
                    std::cerr << "  Seq x1: " << results_sequential[i].x1[j] << ", Par x1: " << results_parallel[i].x1[j] << std::endl;
                    std::cerr << "  Seq x2: " << results_sequential[i].x2[j] << ", Par x2: " << results_parallel[i].x2[j] << std::endl;
                    break;
                }
            }
            if (!all_match) break;
        }
    } else {
        all_match = false;
        std::cerr << "Result vectors size mismatch!" << std::endl;
    }

    if (all_match)
        std::cout << "\nSequential and parallel results match within tolerance." << std::endl;
    else
        std::cout << "\nSequential and parallel results DO NOT match!" << std::endl;

    int q; std::cin >> q;

    return 0;
}
