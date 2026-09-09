// Full-Q multivariate adapter using the authors' unchanged interpolation and
// coefficient-lifting routines. Sampling and independent checks live here.
#include "reconstruction.h"
#include "primes.h"
#include "q_stress_input.hpp"
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <random>

std::pair<std::string,long> thiele(const std::vector<std::string>&,
    const std::vector<std::string>&,std::string,bool);
std::pair<std::string,long> balanced_zippel(std::string&,std::vector<std::string>&,
    std::vector<std::string>&,std::string&,std::map<std::string,std::string>&,bool,bool);
std::pair<std::string,long> rational_reconstruct_multiple(const fmpz*,std::vector<std::string>&,bool);
using U = mp_limb_t;
using Clock = std::chrono::steady_clock;
using fuel::mod_uni_ratfunc_flint::rational_function;

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
static std::unique_ptr<QInput> input;
static std::map<std::vector<U>,U> cache;
static std::map<U,uint64_t> distribution;
static std::map<std::string,size_t> hints;
static Clock::time_point started;
static std::string case_name, seed_name;
static uint64_t probes=0, cap, images=0;
static double timeout;
static bool learned;
void report(const char* status, double elapsed) {
    std::cout << "case,method,seed,status,elapsed_us,probes,primes,images,probes_by_prime\n"
              << case_name << ',' << (learned ? "FIRE7_Q_learned" : "FIRE7_Q") << ','
              << seed_name << ',' << status << ',' << std::fixed << std::setprecision(3)
              << elapsed << ',' << probes << ',' << distribution.size() << ',' << images << ',';
    bool first=true;
    for(auto [p,n] : distribution) {
        if(!first) std::cout << ';';
        first=false;
        std::cout << p << ':' << n;
    }
    std::cout << std::endl;
}
U oracle(const std::vector<U>& key) {
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-started).count();
    if(elapsed>timeout*1e6 || probes>=cap) {
        report(probes>=cap ? "probe_limit" : "time_limit",elapsed);
        std::_Exit(0);
    }
    ++probes; ++distribution[prime];
    return cache[key]=input->evaluate(std::vector<Mod>(key.begin(),key.end())).n;
}
void set_prime(U p) {
    prime=flint_prime=p;
    nmod_init(&flint_mod,p);
    input->set_prime(p);
    cache.clear();
}
std::string slice(const std::string& var,U offset,const std::function<U(U)>& eval) {
    rational_function::initialize_vars({var},prime);
    std::vector<std::string> points,values;
    size_t hint=learned && hints.count(var) ? hints[var] : 8;
    for(size_t count=hint;count<=2048;count*=2) {
        while(points.size()<count) {
            U x=offset+points.size();
            points.push_back(std::to_string(x)); values.push_back(std::to_string(eval(x)));
        }
        auto result=thiele(points,values,var,true);
        if(result.second>0) {
            hints[var]=result.second+1;
            return result.first;
        }
    }
    throw std::runtime_error("Thiele failed within sample budget");
}
std::vector<const char*> variable_names() {
    std::vector<const char*> names;
    for(const auto& name : input->names) names.push_back(name.c_str());
    return names;
}
std::vector<U> random_point(std::mt19937_64& rng) {
    std::uniform_int_distribution<U> random(1,prime-4096);
    std::vector<U> point(input->variables);
    for(auto& x : point) x=random(rng);
    return point;
}
std::string modular_image(std::mt19937_64& rng) {
    const auto nv=input->variables;
    std::uniform_int_distribution<U> random(1,prime-4096);
    // Preserve the former bivariate adapter's a,b,offset draw order.
    std::vector<U> bases(nv-1), anchors(nv,1);
    for(auto& x : bases) x=random(rng);
    for(size_t i=1;i<nv;++i) anchors[i]=random(rng);
    U offset=random(rng);
    if(!fuel::initialize(input->names,1,true,prime)) throw std::runtime_error("FUEL initialization failed");
    fuel::switchToConventional();
    std::string skeleton=slice(input->names[0],offset,[&](U t){
        auto point=anchors; point[0]=t; return oracle(point);
    });
    auto names=variable_names();
    nmod_mpoly_ctx_t ctx;
    nmod_mpoly_ctx_init(ctx,nv,ORD_LEX,prime);
    nmod_mpoly_t n,d;
    nmod_mpoly_init(n,ctx); nmod_mpoly_init(d,ctx);
    auto parse_result=[&](const std::string& expression) {
        auto nd=numerator_denominator(expression);
        if(nmod_mpoly_set_str_pretty(n,nd.first.c_str(),names.data(),ctx) ||
           nmod_mpoly_set_str_pretty(d,nd.second.c_str(),names.data(),ctx))
            throw std::runtime_error("invalid modular expression");
    };
    for(size_t variable=1;variable<nv;++variable) {
        parse_result(skeleton);
        size_t rows=std::max(nmod_mpoly_length(n,ctx),nmod_mpoly_length(d,ctx));
        std::vector<std::string> points,values;
        unsigned degree=0;
        std::string symbol=input->names[variable];
        for(size_t i=1;i<=rows;++i) {
            auto point=anchors;
            for(size_t j=0;j<variable;++j) point[j]=nmod_pow_ui(bases[j],i,flint_mod);
            values.push_back(slice(symbol,offset,[&](U t){
                point[variable]=t; return oracle(point);
            }));
            points.push_back(std::to_string(i));
            auto row=numerator_denominator(values.back());
            degree=std::max(degree,static_cast<unsigned>(std::max(exponent(row.first,symbol),exponent(row.second,symbol))));
        }
        skel_var_value=std::to_string(anchors[variable]); skel_var_power=degree+3;
        std::map<std::string,std::string> balancing;
        for(size_t j=0;j<variable;++j) balancing.emplace(input->names[j],std::to_string(bases[j]));
        auto result=balanced_zippel(skeleton,points,values,symbol,balancing,true,true);
        if(result.second<0) throw std::runtime_error("balanced Zippel failed");
        skeleton=std::move(result.first);
    }
    parse_result(skeleton);
    for(size_t i=0;i<3;++i) {
        auto p=random_point(rng);
        U num=nmod_mpoly_evaluate_all_ui(n,p.data(),ctx),den=nmod_mpoly_evaluate_all_ui(d,p.data(),ctx);
        if(!den || num!=nmod_mul(oracle(p),den,flint_mod)) throw std::runtime_error("modular validation failed");
    }
    nmod_mpoly_clear(n,ctx); nmod_mpoly_clear(d,ctx); nmod_mpoly_ctx_clear(ctx);
    return skeleton;
}
bool validate_candidate(const std::string& candidate,std::mt19937_64& rng) {
    auto nd=numerator_denominator(candidate);
    auto names=variable_names();
    nmod_mpoly_ctx_t ctx; nmod_mpoly_ctx_init(ctx,input->variables,ORD_LEX,prime);
    nmod_mpoly_t n,d; nmod_mpoly_init(n,ctx); nmod_mpoly_init(d,ctx);
    if(nmod_mpoly_set_str_pretty(n,nd.first.c_str(),names.data(),ctx) ||
       nmod_mpoly_set_str_pretty(d,nd.second.c_str(),names.data(),ctx)) throw std::runtime_error("invalid Q candidate");
    bool ok=true;
    for(size_t i=0;i<3;++i) {
        auto p=random_point(rng);
        U num=nmod_mpoly_evaluate_all_ui(n,p.data(),ctx),den=nmod_mpoly_evaluate_all_ui(d,p.data(),ctx);
        if(!den || num!=nmod_mul(oracle(p),den,flint_mod)) {ok=false; break;}
    }
    nmod_mpoly_clear(n,ctx); nmod_mpoly_clear(d,ctx); nmod_mpoly_ctx_clear(ctx);
    return ok;
}
int main(int argc,char** argv) {
    if(argc!=5) throw std::runtime_error("usage: fire7-q-stress ORACLE_FILE CASE SEED default|learned");
    input=std::make_unique<QInput>(argv[1]);
    // The upstream coefficient-lifting routine uses fixed 16-entry exponent buffers.
    if(input->variables>16) throw std::runtime_error("FIRE7 Q adapter supports at most 16 variables");
    case_name=argv[2]; seed_name=argv[3]; learned=std::string(argv[4])=="learned";
    if(!learned && std::string(argv[4])!="default") throw std::runtime_error("unknown mode");
    timeout=std::getenv("BENCH_TIMEOUT") ? std::stod(std::getenv("BENCH_TIMEOUT")) : 180.;
    cap=std::getenv("MAX_TOTAL_PROBES") ? std::stoull(std::getenv("MAX_TOTAL_PROBES")) : 2000000;
    size_t max_primes=std::getenv("MAX_PRIMES") ? std::stoul(std::getenv("MAX_PRIMES")) : 32;
    if(max_primes<2 || max_primes>127) throw std::runtime_error("MAX_PRIMES must be in 2..127");
    std::mt19937_64 rng(std::stoull(seed_name));
    std::vector<std::string> values;
    fmpz* points=_fmpz_vec_init(max_primes);
    std::string candidate;
    started=Clock::now(); fuel::setLibrary("flint");
    for(size_t i=0;i<max_primes;++i) {
        set_prime(primes[i+1]);
        if(!candidate.empty() && validate_candidate(candidate,rng)) {
            double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-started).count();
            check_q_identity(*input,candidate);
            report("ok",elapsed); _fmpz_vec_clear(points,max_primes); return 0;
        }
        values.push_back(modular_image(rng)); ++images;
        fmpz_set_ui(points+i,prime);
        // The authors' routine consumes its strings and uses the global vars.
        // Preserve each reconstructed image for the next prime's CRT attempt.
        vars=input->names;
        auto copy=values;
        auto result=rational_reconstruct_multiple(points,copy,true);
        candidate=result.second>=0 ? result.first : "";
    }
    report("prime_limit",std::chrono::duration<double,std::micro>(Clock::now()-started).count());
    _fmpz_vec_clear(points,max_primes);
}
