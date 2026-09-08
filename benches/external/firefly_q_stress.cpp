// Exact integer-coefficient input shared with Symbolica's Q stress benchmark.
#include "firefly/Reconstructor.hpp"
#include <flint/fmpq_mpoly.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>

using Clock = std::chrono::steady_clock;
struct QTerm { mpz_class integer; uint64_t residue; std::vector<ulong> exponents; };
struct QInput {
    size_t variables;
    uint64_t prime = 0;
    std::vector<std::string> names;
    std::vector<QTerm> numerator, denominator;
    std::vector<size_t> degrees;
    explicit QInput(const std::string& path) {
        std::ifstream in(path);
        size_t ns, ds;
        uint64_t marker;
        if (!(in >> variables >> marker >> ns >> ds) || !variables || marker || !ds)
            throw std::runtime_error("invalid Q oracle header");
        names.resize(variables); degrees.resize(variables);
        for (auto& name : names) in >> name;
        numerator.resize(ns); denominator.resize(ds);
        for (auto* terms : {&numerator, &denominator}) for (auto& t : *terms) {
            in >> t.integer;
            t.exponents.resize(variables);
            for (size_t i=0; i<variables; ++i) {
                in >> t.exponents[i];
                degrees[i] = std::max(degrees[i], size_t(t.exponents[i]));
            }
        }
        if (!in) throw std::runtime_error("truncated Q oracle input");
    }
    void set_prime(uint64_t p) {
        prime=p;
        for (auto* terms : {&numerator, &denominator}) for (auto& t : *terms)
            t.residue=mpz_fdiv_ui(t.integer.get_mpz_t(),prime);
    }
    template<class T> T evaluate(const std::vector<T>& x) const {
        thread_local std::vector<std::vector<T>> powers;
        powers.resize(variables);
        for (size_t i=0;i<variables;++i) {
            powers[i].resize(degrees[i]+1,T(1));
            for (size_t e=1;e<powers[i].size();++e) powers[i][e]=powers[i][e-1]*x[i];
        }
        auto eval = [&](const std::vector<QTerm>& terms) {
            T value(0);
            for (const auto& t : terms) {
                T c(t.residue);
                for (size_t i=0;i<variables;++i) if(t.exponents[i]) c*=powers[i][t.exponents[i]];
                value+=c;
            }
            return value;
        };
        return eval(numerator)/eval(denominator);
    }
};

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

void check(const QInput& input, const firefly::RationalFunction& result) {
    // Serialization restores scanned factors and any internal variable ordering.
    auto expression=result.to_string(input.names);
    size_t split=std::string::npos;
    int depth=0;
    for(size_t i=0;i<expression.size();++i) {
        if(expression[i]=='(') ++depth;
        if(expression[i]==')') --depth;
        if(expression[i]=='/' && depth==0) { split=i; break; }
    }
    if(split==std::string::npos) throw std::runtime_error("invalid rational serialization");
    std::vector<const char*> names;
    for(const auto& name : input.names) names.push_back(name.c_str());
    fmpq_mpoly_ctx_t ctx;
    fmpq_mpoly_ctx_init(ctx,input.variables,ORD_LEX);
    fmpq_mpoly_t n,d,a,b,lhs,rhs;
    for(auto p : {n,d,a,b,lhs,rhs}) fmpq_mpoly_init(p,ctx);
    fmpq_t coefficient;
    fmpq_init(coefficient);
    auto convert = [&](fmpq_mpoly_t out,const std::vector<QTerm>& terms) {
        for(const auto& t : terms) {
            mpq_class c(t.integer);
            fmpq_set_mpq(coefficient,c.get_mpq_t());
            fmpq_mpoly_set_coeff_fmpq_ui(out,coefficient,t.exponents.data(),ctx);
        }
    };
    convert(a,input.numerator); convert(b,input.denominator);
    bool ok=!fmpq_mpoly_set_str_pretty(n,expression.substr(0,split).c_str(),names.data(),ctx)
         && !fmpq_mpoly_set_str_pretty(d,expression.substr(split+1).c_str(),names.data(),ctx);
    fmpq_mpoly_mul(lhs,n,b,ctx); fmpq_mpoly_mul(rhs,d,a,ctx);
    ok=ok && !fmpq_mpoly_is_zero(d,ctx) && fmpq_mpoly_equal(lhs,rhs,ctx);
    fmpq_clear(coefficient);
    for(auto p : {n,d,a,b,lhs,rhs}) fmpq_mpoly_clear(p,ctx);
    fmpq_mpoly_ctx_clear(ctx);
    if(!ok) throw std::runtime_error("exact Q cross-product verification failed");
}

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
    check(*input,result[0]);
    state->report("ok",elapsed);
}
