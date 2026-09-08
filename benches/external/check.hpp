// Exact polynomial cross-product checks, outside all benchmark timers.
#pragma once
#include <flint/fmpq_mpoly.h>
#include <flint/nmod_mpoly.h>
#include <stdexcept>
#include <string>

inline void check(const std::string& num, const std::string& den, bool large, uint64_t prime) {
    const char* names[] = {large ? "y" : "x", large ? "d" : "y"};
    const char* ns = large ? "(d+13)^30*(y^2+9)^7+1" : "x*y+2";
    const char* ds = large ? "(d-4)^29*(y^2-1)^5" : "x*y-2*x+4";
    bool ok;
    if (prime) {
        nmod_mpoly_ctx_t ctx;
        nmod_mpoly_ctx_init(ctx, 2, ORD_LEX, prime);
        nmod_mpoly_t n, d, a, b, lhs, rhs;
        for (auto p : {n,d,a,b,lhs,rhs}) nmod_mpoly_init(p,ctx);
        ok = !nmod_mpoly_set_str_pretty(n,num.c_str(),names,ctx)
          && !nmod_mpoly_set_str_pretty(d,den.c_str(),names,ctx)
          && !nmod_mpoly_set_str_pretty(a,ns,names,ctx)
          && !nmod_mpoly_set_str_pretty(b,ds,names,ctx);
        nmod_mpoly_mul(lhs,n,b,ctx); nmod_mpoly_mul(rhs,d,a,ctx);
        ok = ok && !nmod_mpoly_is_zero(d,ctx) && nmod_mpoly_equal(lhs,rhs,ctx);
        for (auto p : {n,d,a,b,lhs,rhs}) nmod_mpoly_clear(p,ctx);
        nmod_mpoly_ctx_clear(ctx);
    } else {
        fmpq_mpoly_ctx_t ctx;
        fmpq_mpoly_ctx_init(ctx, 2, ORD_LEX);
        fmpq_mpoly_t n, d, a, b, lhs, rhs;
        for (auto p : {n,d,a,b,lhs,rhs}) fmpq_mpoly_init(p,ctx);
        ok = !fmpq_mpoly_set_str_pretty(n,num.c_str(),names,ctx)
          && !fmpq_mpoly_set_str_pretty(d,den.c_str(),names,ctx)
          && !fmpq_mpoly_set_str_pretty(a,ns,names,ctx)
          && !fmpq_mpoly_set_str_pretty(b,ds,names,ctx);
        fmpq_mpoly_mul(lhs,n,b,ctx); fmpq_mpoly_mul(rhs,d,a,ctx);
        ok = ok && !fmpq_mpoly_is_zero(d,ctx) && fmpq_mpoly_equal(lhs,rhs,ctx);
        for (auto p : {n,d,a,b,lhs,rhs}) fmpq_mpoly_clear(p,ctx);
        fmpq_mpoly_ctx_clear(ctx);
    }
    if (!ok) throw std::runtime_error("exact cross-product verification failed");
}
