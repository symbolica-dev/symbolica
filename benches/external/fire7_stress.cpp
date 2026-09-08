// Black-box adapter using the authors' unmodified Thiele and balanced-Zippel
// implementations. Sampling/batching and validation belong to this adapter.
#include "reconstruction.h"
#include "stress_input.hpp"
#include <memory>
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>
#include <random>

std::pair<std::string,long> thiele(const std::vector<std::string>&,
    const std::vector<std::string>&,std::string,bool);
std::pair<std::string,long> balanced_zippel(std::string&,std::vector<std::string>&,
    std::vector<std::string>&,std::string&,std::map<std::string,std::string>&,bool,bool);

using fuel::mod_uni_ratfunc_flint::rational_function;
using U = mp_limb_t;
static uint64_t probes = 0;
static std::map<std::pair<U,U>,U> cache;
static std::unique_ptr<Input> input;
static std::chrono::steady_clock::time_point started;
static std::string case_name, seed_name;
static double timeout;
static uint64_t cap;
static bool learned_batches;
static std::map<std::string,size_t> batch_hints;
const char* method_name() { return learned_batches ? "FIRE7_learned_batch" : "FIRE7_balanced_adapter"; }
void report(const char* status) {
    double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    std::cout << "case,method,seed,status,elapsed_us,probes\n"
              << case_name << ',' << method_name() << ',' << seed_name << ',' << status << ','
              << std::fixed << std::setprecision(3) << us << ',' << probes << std::endl;
}
struct Mod {
    U n;
    Mod(U x=0): n(x) {}
    Mod operator*(Mod b) const {return nmod_mul(n,b.n,flint_mod);}
    Mod& operator*=(Mod b) {n=nmod_mul(n,b.n,flint_mod); return *this;}
    Mod& operator+=(Mod b) {n=nmod_add(n,b.n,flint_mod); return *this;}
    Mod operator/(Mod b) const {
        if(!b.n) throw std::runtime_error("pole in benchmark sampling");
        return nmod_div(n,b.n,flint_mod);
    }
};
U oracle(U x,U y) {
    auto key=std::make_pair(x,y);
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>timeout || probes>=cap) {
        report(probes>=cap ? "probe_limit" : "time_limit");
        std::_Exit(0);
    }
    ++probes;
    return cache[key]=input->evaluate(std::vector<Mod>{x,y}).n;
}

std::string slice(const std::string& var, U offset, const std::function<U(U)>& eval) {
    rational_function::initialize_vars({var},prime);
    std::vector<std::string> points,values;
    // Optional learned batches use only the previous row's observed termination.
    size_t hint = learned_batches && batch_hints.count(var) ? batch_hints[var] : 8;
    for (size_t count=hint; count<=2048; count*=2) {
        while(points.size()<count) {
            U x=offset+points.size();
            points.push_back(std::to_string(x)); values.push_back(std::to_string(eval(x)));
        }
        auto result=thiele(points,values,var,true);
        if(result.second>0) {
            batch_hints[var]=result.second+1;
            return result.first;
        }
    }
    throw std::runtime_error("Thiele failed within sample budget");
}

int main(int argc,char** argv) {
    if(argc!=4) throw std::runtime_error("usage: fire7-stress ORACLE_FILE CASE SEED");
    input=std::make_unique<Input>(argv[1]);
    if(input->variables!=2) throw std::runtime_error("FIRE7 stress adapter currently requires two variables");
    case_name=argv[2]; seed_name=argv[3];
    learned_batches=std::getenv("FIRE7_LEARN_BATCH")!=nullptr;
    timeout=std::getenv("BENCH_TIMEOUT") ? std::stod(std::getenv("BENCH_TIMEOUT")) : 120.;
    cap=std::getenv("MAX_PROBES") ? std::stoull(std::getenv("MAX_PROBES")) : 200000;
    U seed=std::stoull(seed_name);
    prime=flint_prime=input->prime;
    nmod_init(&flint_mod,prime);
    std::string x=input->names[0], y=input->names[1];
    const char* names[]={x.c_str(),y.c_str()};
    // Generic field anchors avoid low-integer poles present in IBP coefficients.
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<U> random(1,prime-4096);
    U a=random(rng), b=random(rng), offset=random(rng);
    started=std::chrono::steady_clock::now();
    // FIRE normally performs this setup once for a table reconstruction job.
    fuel::setLibrary("flint");
    if(!fuel::initialize({x,y},1,true,prime)) throw std::runtime_error("FUEL initialization failed");
    fuel::switchToConventional();
    std::string skeleton=slice(x,offset,[&](U t){return oracle(t,b);});
    auto nd=numerator_denominator(skeleton);
    nmod_mpoly_ctx_t ctx;
    nmod_mpoly_ctx_init(ctx,2,ORD_LEX,prime);
    nmod_mpoly_t n,d;
    nmod_mpoly_init(n,ctx); nmod_mpoly_init(d,ctx);
    if(nmod_mpoly_set_str_pretty(n,nd.first.c_str(),names,ctx) ||
       nmod_mpoly_set_str_pretty(d,nd.second.c_str(),names,ctx)) throw std::runtime_error("invalid skeleton");
    size_t rows=std::max(nmod_mpoly_length(n,ctx),nmod_mpoly_length(d,ctx));
    std::vector<std::string> points,values;
    unsigned degree=0;
    for(size_t i=1;i<=rows;++i) {
        U xi=nmod_pow_ui(a,i,flint_mod);
        values.push_back(slice(y,offset,[&](U t){return oracle(xi,t);}));
        points.push_back(std::to_string(i));
        auto row=numerator_denominator(values.back());
        degree=std::max(degree,static_cast<unsigned>(std::max(exponent(row.first,y),exponent(row.second,y))));
    }
    skel_var_value=std::to_string(b);
    skel_var_power=degree+3;
    std::map<std::string,std::string> balancing={{x,std::to_string(a)}};
    auto result=balanced_zippel(skeleton,points,values,y,balancing,true,true);
    if(result.second<0) throw std::runtime_error("balanced Zippel failed");
    nd=numerator_denominator(result.first);
    if(nmod_mpoly_set_str_pretty(n,nd.first.c_str(),names,ctx) ||
       nmod_mpoly_set_str_pretty(d,nd.second.c_str(),names,ctx)) throw std::runtime_error("invalid result");
    for(U i=0;i<3;++i) {
        U p[]={1009+seed+23*i,2003+seed+31*i};
        U nv=nmod_mpoly_evaluate_all_ui(n,p,ctx),dv=nmod_mpoly_evaluate_all_ui(d,p,ctx);
        if(!dv || nv!=nmod_mul(oracle(p[0],p[1]),dv,flint_mod)) throw std::runtime_error("fresh validation failed");
    }
    double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    nmod_mpoly_t original_n, original_d, lhs, rhs;
    for(auto p : {original_n,original_d,lhs,rhs}) nmod_mpoly_init(p,ctx);
    for(const auto& t : input->numerator) nmod_mpoly_set_coeff_ui_ui(original_n,t.coefficient,t.exponents.data(),ctx);
    for(const auto& t : input->denominator) nmod_mpoly_set_coeff_ui_ui(original_d,t.coefficient,t.exponents.data(),ctx);
    nmod_mpoly_mul(lhs,n,original_d,ctx); nmod_mpoly_mul(rhs,d,original_n,ctx);
    if(nmod_mpoly_is_zero(d,ctx) || !nmod_mpoly_equal(lhs,rhs,ctx)) throw std::runtime_error("exact cross-product verification failed");
    for(auto p : {n,d,original_n,original_d,lhs,rhs}) nmod_mpoly_clear(p,ctx);
    nmod_mpoly_ctx_clear(ctx);
    std::cout << "case,method,seed,status,elapsed_us,probes\n"
              << case_name << ',' << method_name() << ',' << seed_name << ",ok,"
              << std::fixed << std::setprecision(3) << elapsed << ',' << probes << '\n';
}
