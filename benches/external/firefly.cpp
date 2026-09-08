#include "firefly/Reconstructor.hpp"
#include "check.hpp"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>

struct BlackBox : firefly::BlackBoxBase<BlackBox> {
    bool large;
    std::shared_ptr<std::atomic<uint64_t>> probes;
    BlackBox(bool l) : large(l), probes(std::make_shared<std::atomic<uint64_t>>(0)) {}
    template<class T> std::vector<T> operator()(const std::vector<T>& x) {
        ++*probes;
        if (large) return {((x[1]+T(13)).pow(30)*(x[0].pow(2)+T(9)).pow(7)+T(1)) /
                           ((x[1]-T(4)).pow(29)*(x[0].pow(2)-T(1)).pow(5))};
        const T xy = x[0]*x[1];
        return {(xy+T(2))/(xy-T(2)*x[0]+T(4))};
    }
    void prime_changed() {}
};

// One independent run per process avoids persistent static reconstruction state.
int main(int argc, char** argv) {
    if (argc != 5) throw std::runtime_error("usage: firefly-bench eq3|eq28 ff|q default|scan seed");
    const bool large = std::string(argv[1]) == "eq28";
    const bool full_q = std::string(argv[2]) == "q";
    const bool scan = std::string(argv[3]) == "scan";
    if (scan && !full_q) throw std::runtime_error("factor-scan comparison requires the full Q result");
    const auto seed = std::stoull(argv[4]);
    BlackBox bb(large);
    const auto start = std::chrono::steady_clock::now();
    firefly::Reconstructor<BlackBox> rec(2,1,1,bb,firefly::Reconstructor<BlackBox>::SILENT);
    firefly::BaseReconst().set_seed(seed);
    if (scan) { rec.enable_shift_scan(); rec.enable_factor_scan(); }
    rec.reconstruct(full_q ? 300 : 1);
    const std::vector<std::string> names = large ? std::vector<std::string>{"y","d"} : std::vector<std::string>{"x","y"};
    // Retrieve the result inside the timer; conversion to strings and checking outside.
    std::string n,d;
    double elapsed;
    if (full_q) {
        auto results = rec.get_result();
        elapsed = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        if (results.size()!=1) throw std::runtime_error("no reconstruction result");
        const auto expression=results[0].to_string(names); // Includes scanned factors and variable reordering.
        int depth=0;
        for(size_t i=0;i<expression.size();++i) {
            if(expression[i]=='(') ++depth;
            if(expression[i]==')') --depth;
            if(expression[i]=='/' && depth==0) {
                n=expression.substr(0,i); d=expression.substr(i+1); break;
            }
        }
        if(n.empty() || d.empty()) throw std::runtime_error("invalid rational function serialization");
    } else {
        auto results = rec.get_result_ff();
        elapsed = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        if (results.size()!=1) throw std::runtime_error("no reconstruction result");
        n=results[0].numerator.to_string(names); d=results[0].denominator.to_string(names);
    }
    check(n,d,large,full_q ? 0 : firefly::primes()[0]);
    std::cout << (large ? "paper_eq28_y_d" : "paper_eq3") << ",FireFly_" << argv[3] << ',' << argv[2] << ',' << seed << ','
              << std::fixed << std::setprecision(3) << elapsed << ',' << *bb.probes << '\n';
}
