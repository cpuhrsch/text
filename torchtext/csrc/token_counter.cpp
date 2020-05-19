#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>


#include <pybind11/pybind11.h>



// namespace torchtext {
// 
// 
// static auto registry =
//     torch::RegisterOperators().op("torchtext::count_tokens", &count_tokens);
// }

std::unordered_map<std::string, int64_t>
count_tokens(const std::vector<std::string> &tokens) {
  std::unordered_map<std::string, int64_t> counter;
  for (const std::string &token : tokens) {
    // std::cout << "token: " << token << std::endl;
    if (counter.count(token)) {
      counter[token] += 1;
    } else {
      counter[token] = 1;
    }
  }
  return counter;
}

// int add(int i, int j) {
//     return i + j;
// }

PYBIND11_MODULE(_torchtext, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
