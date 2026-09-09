// POSIX loader for the shared, single-threaded benchmark trace ABI.
#pragma once
#include <dlfcn.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct TraceOracle {
    void* library = nullptr;
    void* handle = nullptr;
    void (*close)(void*) = nullptr;
    int (*eval)(void*, uint64_t, const uint64_t*, size_t, uint64_t*) = nullptr;
    int (*eval_many)(void*, uint64_t, const uint64_t*, size_t, uint64_t*, size_t) = nullptr;
    size_t output_count = 1;
    uint64_t (*count)(void*) = nullptr;
    template<class T> T symbol(const char* name) {
        void* result = dlsym(library, name);
        if (!result) throw std::runtime_error(dlerror());
        return reinterpret_cast<T>(result);
    }
    explicit TraceOracle(const std::vector<std::string>& names, const std::vector<std::string>& outputs = {}) {
        const char* path = std::getenv("TRACE_ORACLE_PATH");
        const char* lib = std::getenv("TRACE_ORACLE_LIBRARY");
        if (!path || !lib) throw std::runtime_error("trace path and library are required");
        try {
            library = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
            if (!library) throw std::runtime_error(dlerror());
            auto open = symbol<void*(*)(const char*)>(outputs.empty() ? "rr_open" : "rr_open_many");
            close = symbol<void(*)(void*)>("rr_close");
            eval = symbol<decltype(eval)>("rr_evaluate");
            count = symbol<decltype(count)>("rr_calls");
            auto inputs = symbol<size_t(*)(void*)>("rr_inputs");
            auto name = symbol<const char*(*)(void*, size_t)>("rr_input_name");
            handle = open(path);
            if (!handle || inputs(handle) != names.size()) throw std::runtime_error("invalid trace dimensions");
            for (size_t i=0; i<names.size(); ++i) {
                const char* actual = name(handle,i);
                if (!actual || names[i] != actual) throw std::runtime_error("trace variable order mismatch");
            }
            if (!outputs.empty()) {
                eval_many = symbol<decltype(eval_many)>("rr_evaluate_many");
                auto size = symbol<size_t(*)(void*)>("rr_outputs");
                auto output_name = symbol<const char*(*)(void*, size_t)>("rr_output_name");
                output_count = size(handle);
                if (output_count != outputs.size()) throw std::runtime_error("trace output count mismatch");
                for (size_t i=0; i<outputs.size(); ++i) {
                    const char* actual = output_name(handle,i);
                    if (!actual || outputs[i] != actual) throw std::runtime_error("trace output order mismatch");
                }
            }
        } catch (...) {
            if (handle && close) close(handle);
            if (library) dlclose(library);
            throw;
        }
    }
    ~TraceOracle() { if (handle) close(handle); if (library) dlclose(library); }
    TraceOracle(const TraceOracle&) = delete;
    uint64_t evaluate(uint64_t prime, const std::vector<uint64_t>& point) const {
        uint64_t output = 0;
        int status = eval(handle,prime,point.data(),point.size(),&output);
        if (status == 1) throw std::domain_error("undefined inverse in Ratracer trace");
        if (status) throw std::runtime_error("Ratracer interpreter error or unsupported prime");
        return output;
    }
    uint64_t calls() const { return count(handle); }
    std::vector<uint64_t> evaluate_many(uint64_t prime, const std::vector<uint64_t>& point) const {
        if (!eval_many) throw std::runtime_error("multi-output trace is required");
        std::vector<uint64_t> output(output_count);
        int status = eval_many(handle,prime,point.data(),point.size(),output.data(),output.size());
        if (status == 1) throw std::domain_error("undefined inverse in Ratracer trace");
        if (status) throw std::runtime_error("Ratracer interpreter error or unsupported prime");
        for (auto value : output) if (value >= prime) throw std::runtime_error("noncanonical trace output");
        return output;
    }
};
