// Shared sparse polynomial oracle for the external benchmark adapters.
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Term { uint64_t coefficient; std::vector<ulong> exponents; };
struct Input {
    uint64_t prime;
    size_t variables;
    std::vector<std::string> names;
    std::vector<Term> numerator, denominator;
    std::vector<size_t> degrees;
    explicit Input(const std::string& path) {
        std::ifstream in(path);
        size_t ns, ds;
        if (!(in >> variables >> prime >> ns >> ds) || !variables || !ds)
            throw std::runtime_error("invalid oracle header");
        names.resize(variables); degrees.resize(variables);
        for (auto& name : names) in >> name;
        numerator.resize(ns); denominator.resize(ds);
        for (auto* terms : {&numerator, &denominator}) for (auto& t : *terms) {
            in >> t.coefficient;
            if (t.coefficient >= prime) throw std::runtime_error("invalid coefficient");
            t.exponents.resize(variables);
            for (size_t i=0;i<variables;++i) {
                in >> t.exponents[i];
                degrees[i]=std::max(degrees[i],size_t(t.exponents[i]));
            }
        }
        if (!in) throw std::runtime_error("truncated oracle input");
    }
    template<class T> T evaluate(const std::vector<T>& x) const {
        thread_local std::vector<std::vector<T>> powers;
        powers.resize(variables);
        for (size_t i=0;i<variables;++i) {
            powers[i].resize(degrees[i]+1,T(1));
            for (size_t e=1;e<powers[i].size();++e) powers[i][e]=powers[i][e-1]*x[i];
        }
        auto eval = [&](const std::vector<Term>& terms) {
            T result(0);
            for (const auto& t : terms) {
                T c(t.coefficient);
                for (size_t i=0;i<variables;++i)
                    if(t.exponents[i]) c*=powers[i][t.exponents[i]];
                result+=c;
            }
            return result;
        };
        return eval(numerator)/eval(denominator);
    }
};
