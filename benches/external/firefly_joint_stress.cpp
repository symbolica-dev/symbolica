// FireFly's native multi-output reconstruction against the shared trace ABI.
#include "firefly/Reconstructor.hpp"
#include "q_stress_input.hpp"
#include "trace_oracle.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
using Clock = std::chrono::steady_clock;
struct State {
    uint64_t probes=0, cap=2000000;
    double timeout=570;
    Clock::time_point start;
    std::string seed, method;
};
struct JointBox : firefly::BlackBoxBase<JointBox> {
    std::shared_ptr<TraceOracle> trace;
    std::shared_ptr<State> state;
    JointBox(std::shared_ptr<TraceOracle> t, std::shared_ptr<State> s) : trace(t),state(s) {}
    [[noreturn]] void stop(const char* status) const {
        if (trace->calls()!=state->probes) std::abort();
        double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
        std::cout << "method,seed,status,elapsed_us,probes,outputs,completed\n" << state->method << ',' << state->seed << ',' << status << ',' << std::fixed << std::setprecision(3) << elapsed << ',' << state->probes << ',' << trace->output_count << ",0" << std::endl;
        // FireFly invokes the callback on a worker. Exceptions cannot safely
        // propagate through that worker, so flush the failure record here.
        std::_Exit(0);
    }
    std::vector<firefly::FFInt> operator()(const std::vector<firefly::FFInt>& x) {
        if (std::chrono::duration<double>(Clock::now()-state->start).count()>state->timeout) stop("time_limit");
        if (state->probes>=state->cap) stop("probe_limit");
        std::vector<uint64_t> point;
        for (const auto& v : x) point.push_back(v.n);
        ++state->probes;
        std::vector<uint64_t> raw;
        try { raw=trace->evaluate_many(firefly::FFInt::p,point); }
        catch(const std::domain_error&) { stop("trace_pole"); }
        catch(const std::exception&) { stop("trace_error"); }
        std::vector<firefly::FFInt> result;
        for (auto v : raw) result.emplace_back(v);
        return result;
    }
    template<int N> std::vector<firefly::FFIntVec<N>> operator()(const std::vector<firefly::FFIntVec<N>>& x) {
        std::vector<firefly::FFIntVec<N>> result(trace->output_count);
        std::vector<firefly::FFInt> point(x.size());
        for (int lane=0; lane<N; ++lane) {
            for (size_t i=0; i<x.size(); ++i) point[i]=x[i].vec[lane];
            auto value=(*this)(point);
            for (size_t i=0; i<value.size(); ++i) result[i].vec[lane]=value[i];
        }
        return result;
    }
    void prime_changed() {}
};
int main(int argc, char** argv) {
    if(argc!=5) throw std::runtime_error("usage: firefly-joint-stress ORACLE_DIR CASE_LIST SEED default|scan");
    std::ifstream list(argv[2]);
    if(!list) throw std::runtime_error("cannot read case list");
    std::vector<std::string> cases;
    std::vector<std::unique_ptr<QInput>> inputs;
    for(std::string name; std::getline(list,name);) {
        if(name.empty()) throw std::runtime_error("empty case");
        cases.push_back(name);
        inputs.emplace_back(std::make_unique<QInput>(std::string(argv[1])+"/"+name+".q-oracle"));
        if(inputs.back()->names!=inputs[0]->names) throw std::runtime_error("inconsistent variable order");
    }
    if(inputs.empty()) throw std::runtime_error("empty case list");
    const bool scan=std::string(argv[4])=="scan";
    if(!scan && std::string(argv[4])!="default") throw std::runtime_error("unknown mode");
    auto trace=std::make_shared<TraceOracle>(inputs[0]->names,cases);
    auto state=std::make_shared<State>();
    state->seed=argv[3]; state->method=std::string("FireFly_joint_")+argv[4];
    if(auto p=std::getenv("BENCH_TIMEOUT")) state->timeout=std::stod(p);
    if(auto p=std::getenv("MAX_TOTAL_PROBES")) state->cap=std::stoull(p);
    JointBox bb(trace,state);
    setenv("FIREFLY_BENCH_SEED",argv[3],1);
    std::vector<firefly::RationalFunction> result;
    std::string status="ok";
    state->start=Clock::now();
    {
        firefly::Reconstructor<JointBox> rec(inputs[0]->variables,1,1,bb,firefly::Reconstructor<JointBox>::SILENT);
        firefly::BaseReconst().set_seed(std::stoull(argv[3]));
        if(scan) { rec.enable_shift_scan(); rec.enable_factor_scan(); }
        rec.reconstruct(std::getenv("MAX_PRIMES") ? std::stoul(std::getenv("MAX_PRIMES")) : 32);
        result=rec.get_result();
        if(result.size()!=cases.size()) status="prime_limit";
    }
    double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
    if(trace->calls()!=state->probes) throw std::runtime_error("trace probe accounting mismatch");
    if(status=="ok") for(size_t i=0;i<result.size();++i) check_q_identity(*inputs[i],result[i].to_string(inputs[i]->names));
    std::cout << "method,seed,status,elapsed_us,probes,outputs,completed\nFireFly_joint_" << argv[4] << ',' << argv[3] << ',' << status << ',' << std::fixed << std::setprecision(3) << elapsed << ',' << state->probes << ',' << cases.size() << ',' << result.size() << std::endl;
}
