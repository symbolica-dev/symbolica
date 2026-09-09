// Exact integer-coefficient input shared with Symbolica's Q stress benchmark.
#include "firefly/Reconstructor.hpp"
#include "q_stress_input.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>

using Clock = std::chrono::steady_clock;

struct State {
    uint64_t probes=0, cap;
    std::map<uint64_t,uint64_t> distribution;
    Clock::time_point start;
    double timeout;
    std::string case_name, seed, method;
    void report(const char* status, double elapsed) const {
        std::cout << "case,method,seed,status,elapsed_us,probes,primes,images,probes_by_prime\n"
                  << case_name << ',' << method << ',' << seed << ',' << status << ','
                  << std::fixed << std::setprecision(3) << elapsed << ',' << probes << ','
                  << distribution.size() << ",,";
        bool first=true;
        for (auto [p,n] : distribution) {
            if (!first) std::cout << ';';
            first=false;
            std::cout << p << ':' << n;
        }
        std::cout << std::endl;
    }
};
struct BlackBox : firefly::BlackBoxBase<BlackBox> {
    std::shared_ptr<QInput> input;
    std::shared_ptr<State> state;
    BlackBox(std::shared_ptr<QInput> i, std::shared_ptr<State> s) : input(i),state(s) {}
    template<class T> std::vector<T> operator()(const std::vector<T>& x) {
        double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
        if(elapsed > state->timeout*1e6 || state->probes >= state->cap) {
            state->report(elapsed > state->timeout*1e6 ? "time_limit" : "probe_limit",elapsed);
            std::_Exit(0);
        }
        ++state->probes;
        ++state->distribution[input->prime];
        return {input->evaluate(x)};
    }
    void prime_changed() { input->set_prime(firefly::FFInt::p); }
};


int main(int argc,char** argv) {
    if(argc!=5) throw std::runtime_error("usage: firefly-q-stress ORACLE_FILE CASE SEED default|scan");
    auto input=std::make_shared<QInput>(argv[1]);
    auto state=std::make_shared<State>();
    const bool scan=std::string(argv[4])=="scan";
    if(!scan && std::string(argv[4])!="default") throw std::runtime_error("unknown mode");
    state->case_name=argv[2]; state->seed=argv[3];
    state->method=scan ? "FireFly_scan" : "FireFly_default";
    state->timeout=std::getenv("BENCH_TIMEOUT") ? std::stod(std::getenv("BENCH_TIMEOUT")) : 180.;
    state->cap=std::getenv("MAX_TOTAL_PROBES") ? std::stoull(std::getenv("MAX_TOTAL_PROBES")) : 2000000;
    const auto max_primes=std::getenv("MAX_PRIMES") ? std::stoul(std::getenv("MAX_PRIMES")) : 32;
    BlackBox bb(input,state);
    // The pinned constructor reads this before the later explicit set_seed call.
    setenv("FIREFLY_BENCH_SEED",state->seed.c_str(),1);
    state->start=Clock::now();
    firefly::Reconstructor<BlackBox> rec(input->variables,1,1,bb,firefly::Reconstructor<BlackBox>::SILENT);
    firefly::BaseReconst().set_seed(std::stoull(state->seed));
    if(scan) { rec.enable_shift_scan(); rec.enable_factor_scan(); }
    rec.reconstruct(max_primes);
    auto result=rec.get_result();
    double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
    if(result.empty()) { state->report("prime_limit",elapsed); return 0; }
    if(result.size()!=1) throw std::runtime_error("unexpected reconstruction result count");
    // Serialization restores scanned factors and internal variable ordering.
    check_q_identity(*input,result[0].to_string(input->names));
    state->report("ok",elapsed);
}
