#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>


namespace torchtext {

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

static auto registry =
    torch::RegisterOperators().op("torchtext::count_tokens", &count_tokens);
}
