#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <stdexcept>
#include <thread>

using namespace torch::jit;
// namespace torchtext {
// 
// 
// static auto registry =
//     torch::RegisterOperators().op("torchtext::count_tokens", &count_tokens);
// }

void process(
    StrongFunctionPtr sfn,
    std::shared_ptr<const std::vector<std::string>> tokens,
    std::shared_ptr<std::unordered_map<std::string, int64_t>> counter,
    size_t start,
    size_t end) {

  Function &operation = *sfn.function_;
  for (size_t i = start; i < end; i++) {
    auto token_ = tokens->at(i);
    auto token_list =
        operation(std::vector<c10::IValue>({c10::IValue(token_)})).toList();
    for (size_t i = 0; i < token_list.size(); i++) {
      c10::IValue t_ref = token_list.get(i);
      std::string t = t_ref.toStringRef();
      if (counter->count(t)) {
        counter->insert(std::make_pair(t, 1 + counter->at(t)));
      } else {
        counter->insert(std::make_pair(t, 1));
      }
    }
  }
}
std::unordered_map<std::string, int64_t>
count_tokens(py::object fn, const std::vector<std::string> &tokens) {
  if (!py::isinstance<StrongFunctionPtr>(fn)) {
    throw std::runtime_error("asdf");
  }

  auto sfn = py::cast<StrongFunctionPtr>(fn);
  std::vector<std::thread> threads;
  size_t num_cpus = std::thread::hardware_concurrency();
  std::vector<std::unordered_map<std::string, int64_t>> counters(num_cpus);
  size_t chunk_size = tokens.size() / num_cpus;
  auto tokens_ptr = std::make_shared<const std::vector<std::string>>(tokens);
  for (size_t i = 0; i < num_cpus; i++) {
    auto counter_ptr =
        std::make_shared<std::unordered_map<std::string, int64_t>>(counters[i]);
    threads.emplace_back(
        std::thread(process, sfn, tokens_ptr, counter_ptr, i * chunk_size,
                    std::min(((i + 1) * chunk_size), tokens.size())));
  }
  std::unordered_map<std::string, int64_t> counter;
  for (size_t i = 0; i < num_cpus; i++) {
    threads[i].join();
    counter.insert(counters[i].begin(), counters[i].end());
  }
  return counter;
}

// int add(int i, int j) {
//     return i + j;
// }

PYBIND11_MODULE(_torchtext, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &count_tokens, "A function which adds two numbers");
}
