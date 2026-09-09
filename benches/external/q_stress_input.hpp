// Exact sparse integer input and independent Q identity checker.
#pragma once
#include <flint/fmpq_mpoly.h>
#include <gmpxx.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

inline void check_q_identity(const QInput& input, const std::string& expression) {
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
            fmpq_set_str(coefficient,t.integer.get_str().c_str(),10);
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
