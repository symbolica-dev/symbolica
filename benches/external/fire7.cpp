// Black-box adapter using the authors' unmodified Thiele and balanced-Zippel
// implementations. Sampling/batching and validation belong to this adapter.
#include "reconstruction.h"
#include "check.hpp"
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>

std::pair<std::string,long> thiele(const std::vector<std::string>&,
    const std::vector<std::string>&,std::string,bool);
std::pair<std::string,long> balanced_zippel(std::string&,std::vector<std::string>&,
    std::vector<std::string>&,std::string&,std::map<std::string,std::string>&,bool,bool);

using fuel::mod_uni_ratfunc_flint::rational_function;
using U = mp_limb_t;
static uint64_t probes = 0;
static std::map<std::pair<U,U>,U> cache;
static bool large;
U oracle(U x, U y) {
    auto key=std::make_pair(x,y);
    if (cache.count(key)) return cache.at(key);
    U n,d;
    auto mul=[](U a,U b){return nmod_mul(a,b,flint_mod);};
    auto add=[](U a,U b){return nmod_add(a,b,flint_mod);};
    auto sub=[](U a,U b){return nmod_sub(a,b,flint_mod);};
    auto pow=[](U a,U b){return nmod_pow_ui(a,b,flint_mod);};
    if (large) {
        n=add(mul(pow(add(y,13),30),pow(add(pow(x,2),9),7)),1);
        d=mul(pow(sub(y,4),29),pow(sub(pow(x,2),1),5));
    } else { n=add(mul(x,y),2); d=add(sub(mul(x,y),mul(2,x)),4); }
    ++probes;
    if (!d) throw std::runtime_error("pole in benchmark sampling");
    return cache[key]=nmod_div(n,d,flint_mod);
}

std::string slice(const std::string& var, U offset, const std::function<U(U)>& eval) {
    rational_function::initialize_vars({var},prime);
    std::vector<std::string> points,values;
    // No degrees supplied. Reuse evaluations while doubling the input batch.
    for (size_t count=8; count<=256; count*=2) {
        while(points.size()<count) {
            U x=offset+points.size();
            points.push_back(std::to_string(x)); values.push_back(std::to_string(eval(x)));
        }
        auto result=thiele(points,values,var,true);
        if(result.second>0) return result.first;
    }
    throw std::runtime_error("Thiele failed within sample budget");
}

int main(int argc,char** argv) {
    if(argc!=3) throw std::runtime_error("usage: fire7-bench eq3|eq28 seed");
    large=std::string(argv[1])=="eq28";
    U seed=std::stoull(argv[2]);
    prime=flint_prime=9223372036854775783ULL;
    nmod_init(&flint_mod,prime);
    std::string x=large?"y":"x", y=large?"d":"y";
    const char* names[]={x.c_str(),y.c_str()};
    U a=3+seed, b=17+seed, offset=101+seed;
    const auto start=std::chrono::steady_clock::now();
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
    double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
    check(nd.first,nd.second,large,prime);
    nmod_mpoly_clear(n,ctx); nmod_mpoly_clear(d,ctx); nmod_mpoly_ctx_clear(ctx);
    std::cout<<(large?"paper_eq28_y_d":"paper_eq3")<<",FIRE7_balanced_adapter,ff,"<<seed<<','
             <<std::fixed<<std::setprecision(3)<<elapsed<<','<<probes<<'\n';
}
