#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <stdexcept>
#include <thread>

using namespace torch::jit;

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
count_tokens(py::object fn, const std::vector<std::string> &tokens, size_t num_cpus) {
  if (!py::isinstance<StrongFunctionPtr>(fn)) {
    throw std::runtime_error("Given object is not a JIT function.");
  }

  std::unordered_map<std::string, int64_t> counter;
  auto sfn = py::cast<StrongFunctionPtr>(fn);
  std::vector<std::thread> threads;
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
  for (size_t i = 0; i < num_cpus; i++) {
    threads[i].join();
    counter.insert(counters[i].begin(), counters[i].end());
  }
  return counter;
}

PYBIND11_MODULE(_torchtext, m) {
    m.def("count_tokens", &count_tokens,
          "Apply JIT'd function to a list of strings and count number of "
          "resulting tokens");
}
