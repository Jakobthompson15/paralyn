#include "paralyn/native.h"
#include "paralyn/executable.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned negative_checks = 0;
std::vector<pr_event_info> events;
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
void success(pr_status status) {
  if (status == PR_SUCCESS) return;
  pr_error error{}; pr_last_error(&error);
  throw std::runtime_error(std::string(error.operation) + ": " + error.message);
}
void failure(pr_status status, pr_status expected, const char *operation) {
  pr_error error{}; pr_last_error(&error);
  require(status == expected && error.code == expected && error.operation == std::string(operation) && *error.message,
          std::string("Incorrect failure for ") + operation + ": " + std::to_string(status) + " " + error.message);
  ++negative_checks;
}
struct Handles {
  std::vector<pr_handle> values;
  pr_handle keep(pr_handle h) { require(h != 0, "Invalid successful handle"); values.push_back(h); return h; }
  void release(pr_handle h) {
    for (auto &v : values) if (v == h) { v = 0; success(pr_release(h)); return; }
    throw std::runtime_error("Missing owned handle");
  }
  void close() { for (auto i = values.rbegin(); i != values.rend(); ++i) if (*i) { const auto h = *i; *i = 0; success(pr_release(h)); } }
  ~Handles() { for (auto i = values.rbegin(); i != values.rend(); ++i) if (*i) pr_release(*i); }
};
pr_argument buffer(pr_view v) { pr_argument a{}; a.type = PR_BUFFER; a.view = v; return a; }
pr_argument integer(std::uint32_t n) { pr_argument a{}; a.type = PR_U32; a.u32 = n; return a; }
pr_buffer allocation(Handles &h, pr_context c, const std::vector<float> &data) {
  pr_buffer b = 0; success(pr_buffer_create(c, data.size()*4, &b)); h.keep(b);
  success(pr_buffer_write(b, 0, data.data(), data.size()*4)); return b;
}
pr_view view(Handles &h, pr_buffer b, std::size_t count, pr_access access) {
  pr_view v = 0; success(pr_view_create(b, 4, count*4, 4, access, &v)); return h.keep(v);
}
pr_event launch(Handles &h, pr_queue q, pr_kernel k, pr_dim3 grid, pr_dim3 block, std::vector<pr_argument> &args) {
  pr_event e = 0; success(pr_launch(q,k,grid,block,args.data(),static_cast<uint32_t>(args.size()),&e)); return h.keep(e);
}
void completed(pr_event e) {
  pr_event_info info{}; success(pr_event_wait(e,&info));
  require(info.completed && info.gpu_start_seconds > 0 && info.gpu_end_seconds >= info.gpu_start_seconds,
          "Missing real completed GPU event/timestamps");
  pr_event_timing_v1 timing{};
  timing.struct_size = sizeof(timing); timing.version = 1;
  success(pr_event_timing(e,&timing));
  require(timing.completed && timing.duration_valid && timing.timestamps_valid &&
          timing.clock_domain == PR_CLOCK_METAL_SYSTEM_MACH && timing.duration_seconds > 0 &&
          timing.start_seconds == info.gpu_start_seconds && timing.end_seconds == info.gpu_end_seconds &&
          timing.duration_seconds == timing.end_seconds - timing.start_seconds,
          "Invalid versioned GPU timing contract");
  events.push_back(info);
}
void read_equal(pr_buffer b, const std::vector<float> &expected) {
  std::vector<float> actual(expected.size()); success(pr_buffer_read(b,0,actual.data(),actual.size()*4));
  for (std::size_t i=0;i<actual.size();++i)
    require(actual[i] == expected[i], "Independent CPU verification mismatch at element " + std::to_string(i));
}
std::vector<float> values(std::size_t n, int seed) {
  std::vector<float> v(n+2, -65536.0f);
  for (std::size_t i=0;i<n;++i) v[i+1] = float((i*17+seed)%127) - 63.0f;
  return v;
}
// Independent encoder deliberately bypasses the production verifier so negative
// tests reach the public loader with malformed descriptors, never GPU dispatch.
std::vector<unsigned char> unchecked(const paralyn::ExecutableModule &m) {
  std::vector<unsigned char> b{'P','A','R','A','L','Y','N','X'};
  auto u32 = [&](std::uint32_t n) { for(unsigned i=0;i<32;i+=8)b.push_back(static_cast<unsigned char>(n>>i)); };
  auto str = [&](const std::string &s) { u32(static_cast<std::uint32_t>(s.size()));b.insert(b.end(),s.begin(),s.end()); };
  u32(1);u32(static_cast<std::uint32_t>(m.format));str(m.target);u32(m.numerical_policy);
  str(m.producer);str(m.producer_version);str(m.source_name);str(m.source_sha256);str(m.source);
  u32(static_cast<std::uint32_t>(m.entries.size()));
  for(const auto &e:m.entries){
    str(e.name);for(auto n:e.required_block)u32(n);u32(static_cast<std::uint32_t>(e.parameters.size()));
    for(const auto &p:e.parameters){
      str(p.name);u32(p.type==paralyn::ScalarType::I32?1:p.type==paralyn::ScalarType::U32?2:3);
      u32(p.buffer);u32(static_cast<std::uint32_t>(p.access));u32(p.binding);u32(p.alignment);
      u32(static_cast<std::uint32_t>(p.minimum_bytes));u32(static_cast<std::uint32_t>(p.minimum_bytes>>32));
    }
  }
  return b;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "Usage: msl_tests MODULE.prx NEW_OUTPUT_DIR");
    const std::filesystem::path output = argv[2];
    require(!std::filesystem::exists(output), "Evidence directory must be new");
    std::ifstream input(argv[1], std::ios::binary);
    require(bool(input), "Cannot open executable module");
    std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), {}};
    const auto descriptor = paralyn::deserialize_executable(bytes.data(), bytes.size());
    Handles global;
    pr_context context = 0; success(pr_context_create("auto", &context)); global.keep(context);
    pr_queue queue = 0; success(pr_queue_get(context,&queue)); global.keep(queue);
    pr_module module = 0; success(pr_module_load_file(context,argv[1],&module)); global.keep(module);
    pr_kernel add = 0, reduce = 0, transpose = 0;
    success(pr_module_kernel(module,"vector_add",&add)); global.keep(add);
    success(pr_module_kernel(module,"block_reduce",&reduce)); global.keep(reduce);
    success(pr_module_kernel(module,"tiled_transpose",&transpose)); global.keep(transpose);
    global.release(module); // Every kernel retains its module and compiled pipelines.
    pr_parameter_info parameter{};
    success(pr_kernel_parameter(add,2,&parameter));
    require(parameter.type == PR_F32 && parameter.is_buffer && parameter.access == PR_READ_WRITE,
            "MSL parameter reflection contract changed");

    auto invalid_module = [&](paralyn::ExecutableModule m) {
      auto bad = unchecked(m);
      pr_module handle = 99;
      failure(pr_module_load(context,bad.data(),bad.size(),&handle),PR_COMPILATION_FAILED,"module_load");
      require(handle == 0, "Failed module leaked a handle");
    };
    auto bad = descriptor; bad.entries[0].name = "missing_entry"; invalid_module(bad);
    bad = descriptor; bad.entries[0].parameters[0].type = paralyn::ScalarType::I32; invalid_module(bad);
    bad = descriptor; bad.entries[0].parameters[0].binding = 30; invalid_module(bad);
    bad = descriptor; bad.entries[0].parameters[2].access = paralyn::ResourceAccess::Read; invalid_module(bad);
    bad = descriptor; bad.entries[0].parameters[0].buffer = false; invalid_module(bad);
    bad = descriptor; bad.entries[0].parameters[3].buffer = true; invalid_module(bad);
    bad = descriptor; bad.source += "\ninvalid MSL tokens\n"; bad.source_sha256 = paralyn::source_sha256(bad.source); invalid_module(bad);
    auto corrupt = bytes; corrupt.back() ^= 128;
    pr_module invalid = 99;
    failure(pr_module_load(context,corrupt.data(),corrupt.size(),&invalid),PR_COMPILATION_FAILED,"module_load");
    require(invalid == 0, "Corrupt module published a handle");

    for (std::uint32_t n : {1u,65u,1003u}) {
      Handles h;
      auto a = values(n,3), b = values(n,29), expected = std::vector<float>(n+2,-65536.0f);
      for (std::size_t i=1;i<=n;++i) expected[i] = a[i]+b[i];
      auto ab = allocation(h,context,a), bb = allocation(h,context,b);
      auto out = allocation(h,context,std::vector<float>(n+2,-65536.0f));
      auto av=view(h,ab,n,PR_READ), bv=view(h,bb,n,PR_READ), ov=view(h,out,n,PR_READ_WRITE);
      std::vector<pr_argument> args{buffer(av),buffer(bv),buffer(ov),integer(n)};
      if (n == 65) {
        pr_event e = 99;
        auto wrong = args; wrong[3].type=PR_I32;
        failure(pr_launch(queue,add,{2,1,1},{64,1,1},wrong.data(),4,&e),PR_INVALID_ARGUMENT,"launch");
        require(e==0,"Rejected launch leaked event");
        wrong=args; wrong[2]=buffer(av);
        failure(pr_launch(queue,add,{2,1,1},{64,1,1},wrong.data(),4,&e),PR_INVALID_ARGUMENT,"launch");
        auto writable_alias=view(h,ab,n,PR_READ_WRITE);
        wrong=args; wrong[2]=buffer(writable_alias);
        failure(pr_launch(queue,add,{2,1,1},{64,1,1},wrong.data(),4,&e),PR_UNSUPPORTED,"launch");
        h.release(writable_alias);
        failure(pr_launch(queue,add,{0,1,1},{64,1,1},args.data(),4,&e),PR_INVALID_ARGUMENT,"launch");
      }
      auto e=launch(h,queue,add,{(n+63)/64,1,1},{64,1,1},args);
      args[3].u32=0; // Scalar argument bytes were captured at submission.
      if (n == 65) {
        completed(e); read_equal(out,expected);
        for (std::size_t i=1;i<=n;++i) expected[i] = a[i]+a[i];
        args[1]=buffer(av); args[3]=integer(n);
        e=launch(h,queue,add,{(n+63)/64,1,1},{64,1,1},args);
      }
      h.release(av); h.release(bv); h.release(ab); h.release(bb);
      completed(e); read_equal(out,expected); h.close();
    }
    global.release(add);
    for (std::uint32_t n : {1u,64u,1003u}) {
      Handles h;
      auto data=values(n,7); const auto groups=(n+63)/64;
      std::vector<float> expected(groups+2,-65536.0f);
      for (unsigned g=0;g<groups;++g) {
        float sum=0; for (unsigned i=g*64;i<std::min(n,(g+1)*64);++i) sum += data[i+1];
        expected[g+1]=sum; // Small integers make any reduction order exact.
      }
      auto in=allocation(h,context,data), out=allocation(h,context,std::vector<float>(groups+2,-65536.0f));
      auto iv=view(h,in,n,PR_READ), ov=view(h,out,groups,PR_READ_WRITE);
      std::vector<pr_argument> args{buffer(iv),buffer(ov),integer(n)};
      if (n==64) {
        pr_event e=99;
        failure(pr_launch(queue,reduce,{groups,1,1},{32,1,1},args.data(),3,&e),PR_INVALID_ARGUMENT,"launch");
        require(e==0,"Wrong block leaked event");
      }
      auto e=launch(h,queue,reduce,{groups,1,1},{64,1,1},args);
      h.release(iv); h.release(in); completed(e); read_equal(out,expected); h.close();
    }
    global.release(reduce);
    for (auto shape : {std::pair<unsigned,unsigned>{1,1},{17,31},{33,19}}) {
      Handles h; const auto width=shape.first,height=shape.second,n=width*height;
      auto data=values(n,11); std::vector<float> expected(n+2,-65536.0f);
      for (unsigned y=0;y<height;++y) for (unsigned x=0;x<width;++x)
        expected[x*height+y+1]=data[y*width+x+1];
      auto in=allocation(h,context,data), out=allocation(h,context,std::vector<float>(n+2,-65536.0f));
      auto iv=view(h,in,n,PR_READ), ov=view(h,out,n,PR_READ_WRITE);
      std::vector<pr_argument> args{buffer(iv),buffer(ov),integer(width),integer(height)};
      auto e=launch(h,queue,transpose,{(width+15)/16,(height+15)/16,1},{16,16,1},args);
      if (width==33) global.release(transpose); // Submitted work retains the compiled executable.
      h.release(iv); h.release(in); completed(e); read_equal(out,expected); h.close();
    }
    require(events.size()==10,"Unexpected GPU event count");
    std::filesystem::create_directories(output);
    success(pr_context_write_evidence(context,(output/"evidence").c_str()));
    std::ofstream verification(output/"verification.json");
    verification << "{\n  \"verification\": \"PASS\",\n  \"gpu_events\": " << events.size()
                 << ",\n  \"negative_checks\": " << negative_checks
                 << ",\n  \"cpu_fallback\": false,\n  \"workloads\": [\"vector_add\", \"block_reduce\", \"tiled_transpose\"],\n  \"events\": [";
    for (std::size_t i=0;i<events.size();++i) {
      if (i) verification << ',';
      verification << std::setprecision(17) << "{\"gpu_start_seconds\":" << events[i].gpu_start_seconds
                   << ",\"gpu_end_seconds\":" << events[i].gpu_end_seconds << '}';
    }
    verification << "]\n}\n"; verification.close(); require(bool(verification),"Cannot write verification evidence");
    global.close();
    std::cout << "Verification: PASS\n10 physical GPU events; " << negative_checks << " structured negative checks\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
