// Shared benchmark oracle: the pinned Ratracer interpreter, without FireFly.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include "ratracer.h"
#include "ratbox.h"

struct Oracle {
    Trace trace = tr_init();
    std::vector<uint8_t> code;
    std::vector<ncoef_t> data;
    nmod_t modulus{};
    uint64_t calls = 0;
    ~Oracle() {
        for (auto& c : trace.constants) fmpz_clear(&c);
        tr_clear(trace);
    }
};

extern "C" {
void* rr_open(const char* path) {
    try {
        auto oracle = std::make_unique<Oracle>();
        FILE* file = fopen(path, "rb");
        if (!file) return nullptr;
        int status = tr_mergeimport_FILE(oracle->trace, file);
        fclose(file);
        tr_flush(oracle->trace);
        if (status || oracle->trace.noutputs != 1 || code_size(oracle->trace.code)
            || oracle->trace.outputs[0] >= oracle->trace.nextloc) return nullptr;
        size_t size = oracle->trace.fincode.filesize;
        // The upstream interpreter requires zero padding after its last page.
        oracle->code.resize(size + CODE_PAGELUFT, 0);
        for (size_t offset = 0; offset < size;) {
            ssize_t n = pread(oracle->trace.fincode.fd, oracle->code.data() + offset, size - offset, offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return nullptr;
            offset += n;
        }
        oracle->data.resize(oracle->trace.nextloc);
        return oracle.release();
    } catch (...) { return nullptr; }
}
void rr_close(void* handle) { delete static_cast<Oracle*>(handle); }
size_t rr_inputs(void* handle) { return static_cast<Oracle*>(handle)->trace.ninputs; }
const char* rr_input_name(void* handle, size_t index) {
    auto& names = static_cast<Oracle*>(handle)->trace.input_names;
    return index < names.size() ? names[index].c_str() : nullptr;
}
uint64_t rr_calls(void* handle) { return static_cast<Oracle*>(handle)->calls; }
// 0: success; 1: undefined intermediate inverse; -1: invalid input/interpreter error.
int rr_evaluate(void* handle, uint64_t prime, const uint64_t* point, size_t count, uint64_t* output) {
    static_assert(sizeof(ncoef_t) == sizeof(uint64_t));
    auto& oracle = *static_cast<Oracle*>(handle);
    if (prime < 3 || prime >= (uint64_t(1) << 63) || !(prime & 1) || count != oracle.trace.ninputs)
        return -1;
    for (size_t i = 0; i < count; ++i) if (point[i] >= prime) return -1;
    if (oracle.modulus.n != prime) nmod_init(&oracle.modulus, prime);
    ++oracle.calls;
    int status = code_evaluate_lo_mem(oracle.code.data(), oracle.trace.fincode.filesize,
        point, oracle.trace.constants.data(), oracle.data.data(), oracle.modulus);
    if (status == 2 || status == 3) return 1;
    if (status) return -1;
    *output = oracle.data[oracle.trace.outputs[0]];
    return 0;
}
}
