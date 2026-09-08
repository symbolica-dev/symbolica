// Public stress inputs exported by reconstruction_stress_benchmark. Only this
// oracle and the independent exact checker see source terms and coefficients.
#include "firefly/Reconstructor.hpp"
#include <flint/nmod_mpoly.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>

using Clock = std::chrono::steady_clock;
#include "stress_input.hpp"

struct State {
    std::atomic<uint64_t> probes{0};
    Clock::time_point start;
    double timeout;
    uint64_t cap;
    std::string case_name, seed;
    void report(const char* status, double elapsed) const {
        std::cout << "case,method,seed,status,elapsed_us,probes\n"
                  << case_name << ",FireFly_default," << seed << ',' << status << ','
                  << std::fixed << std::setprecision(3) << elapsed << ',' << probes << std::endl;
    }
};
struct BlackBox : firefly::BlackBoxBase<BlackBox> {
    std::shared_ptr<Input> input;
    std::shared_ptr<State> state;
    BlackBox(std::shared_ptr<Input> i, std::shared_ptr<State> s) : input(i),state(s) {}
    template<class T> std::vector<T> operator()(const std::vector<T>& x) {
        double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
        if(elapsed > state->timeout*1e6 || state->probes >= state->cap) {
            state->report(elapsed > state->timeout*1e6 ? "time_limit" : "probe_limit",elapsed);
            // Stop all scheduler threads. This is a fresh, single-run benchmark process.
            std::_Exit(0);
        }
        ++state->probes;
        return {input->evaluate(x)};
    }
    void prime_changed() {}
};

void check(const Input& input, const firefly::RationalFunctionFF& result) {
    nmod_mpoly_ctx_t ctx;
    nmod_mpoly_ctx_init(ctx,input.variables,ORD_LEX,input.prime);
    nmod_mpoly_t n,d,a,b,lhs,rhs;
    for(auto p : {n,d,a,b,lhs,rhs}) nmod_mpoly_init(p,ctx);
    for(const auto& t : input.numerator) nmod_mpoly_set_coeff_ui_ui(a,t.coefficient,t.exponents.data(),ctx);
    for(const auto& t : input.denominator) nmod_mpoly_set_coeff_ui_ui(b,t.coefficient,t.exponents.data(),ctx);
    auto convert = [&](nmod_mpoly_t out,const firefly::PolynomialFF& p) {
        for(const auto& [e,c] : p.coefs) {
            std::vector<ulong> exponents(e.begin(),e.end());
            nmod_mpoly_set_coeff_ui_ui(out,c.n,exponents.data(),ctx);
        }
    };
    convert(n,result.numerator); convert(d,result.denominator);
    nmod_mpoly_mul(lhs,n,b,ctx); nmod_mpoly_mul(rhs,d,a,ctx);
    bool ok = !nmod_mpoly_is_zero(d,ctx) && nmod_mpoly_equal(lhs,rhs,ctx);
    for(auto p : {n,d,a,b,lhs,rhs}) nmod_mpoly_clear(p,ctx);
    nmod_mpoly_ctx_clear(ctx);
    if(!ok) throw std::runtime_error("exact cross-product verification failed");
}

int main(int argc,char** argv) {
    if(argc!=4) throw std::runtime_error("usage: firefly-stress ORACLE_FILE CASE SEED");
    auto input=std::make_shared<Input>(argv[1]);
    if(input->prime!=firefly::primes()[0]) throw std::runtime_error("requires FireFly's first prime");
    auto state=std::make_shared<State>();
    state->case_name=argv[2]; state->seed=argv[3];
    state->timeout=std::getenv("BENCH_TIMEOUT") ? std::stod(std::getenv("BENCH_TIMEOUT")) : 120.;
    state->cap=std::getenv("MAX_PROBES") ? std::stoull(std::getenv("MAX_PROBES")) : 200000;
    BlackBox bb(input,state);
    state->start=Clock::now();
    firefly::Reconstructor<BlackBox> rec(input->variables,1,1,bb,firefly::Reconstructor<BlackBox>::SILENT);
    firefly::BaseReconst().set_seed(std::stoull(state->seed));
    rec.reconstruct(1);
    auto result=rec.get_result_ff();
    double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
    if(result.size()!=1) throw std::runtime_error("missing reconstruction result");
    check(*input,result[0]);
    state->report("ok",elapsed);
}
