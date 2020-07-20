#include <double-conversion/double-conversion.h>
#include <double-conversion/ieee.h>
#include <double-conversion/utils.h>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/script.h>

// timing
#include <chrono>
#include <ctime>
#include <ratio>

using c10::Dict;

namespace torchtext {
namespace {

// TODO: Instead of using Dict, could just use Vectors class
typedef Dict<std::string, torch::Tensor> VectorsDict;
typedef std::vector<std::string> StringList;

struct Vectors : torch::CustomClassHolder {
public:
  Dict<std::string, torch::Tensor> stovec_;
  torch::Tensor unk_tensor_;

  explicit Vectors(const Dict<std::string, torch::Tensor> &stovec,
                   const torch::Tensor &unk_tensor)
      : stovec_(stovec), unk_tensor_(unk_tensor) {}

  explicit Vectors(const std::vector<std::string> &tokens,
                   const torch::Tensor &vectors,
                   const torch::Tensor &unk_tensor)
      : unk_tensor_(std::move(unk_tensor)) {
    // guarding against size mismatch of vectors and tokens
    if (static_cast<int>(tokens.size()) != vectors.size(0)) {
      throw std::runtime_error(
          "Mismatching sizes for tokens and vectors. Size of tokens: " +
          std::to_string(tokens.size()) +
          ", size of vectors: " + std::to_string(vectors.size(0)) + ".");
    }

    stovec_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); i++) {
      // tokens should not have any duplicates
      if (stovec_.find(tokens[i]) != stovec_.end()) {
        throw std::runtime_error("Duplicate token found in tokens list: " +
                                 tokens[i]);
      }
      stovec_.insert(std::move(tokens[i]), vectors.select(0, i));
    }
  }

  torch::Tensor __getitem__(const std::string &token) const {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      return item->value();
    }
    return unk_tensor_;
  }

  torch::Tensor lookup_vectors(const std::vector<std::string> &tokens) {
    std::vector<torch::Tensor> vectors;
    for (const std::string &token : tokens) {
      vectors.push_back(__getitem__(token));
    }

    return torch::stack(vectors, 0);
  }

  void __setitem__(const std::string &token, const torch::Tensor &vector) {
    const auto &item = stovec_.find(token);
    if (item != stovec_.end()) {
      item->value() = vector;
    } else {
      stovec_.insert_or_assign(token, vector);
    }
  }

  int64_t __len__() { return stovec_.size(); }
};

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

std::tuple<int64_t, int64_t, int64_t>
_infer_shape(const std::string &file_path, const int64_t delimiter_ascii) {

  int64_t num_header_lines = 0, num_lines = 0, vector_dim = -1;
  std::vector<std::string> vec_str;
  std::string line, word;

  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  while (std::getline(fin, line)) {
    vec_str.clear();
    if (vector_dim == -1) {
      std::istringstream s(line);

      // get rid of the token
      std::getline(s, word, static_cast<char>(delimiter_ascii));

      // we assume entries for vector are always seperated by ' '
      while (std::getline(s, word, ' ')) {
        vec_str.push_back(word);
      }

      // assuming word, [vector] format
      if (vec_str.size() <= 2) {
        num_header_lines++;
      } else if (vec_str.size() > 2) {
        // the header present in some(w2v) formats contains two elements
        vector_dim = vec_str.size();
        num_lines++; // first element read
      }
    } else {
      num_lines++;
    }
  }
  return std::make_tuple(num_lines, num_header_lines, vector_dim);
}

void parse_chunk(const std::string &file_path, const int64_t start_line,
                 const int64_t end_line, const int64_t vector_dim,
                 const int64_t delimiter_ascii,
                 std::shared_ptr<Dict<std::string, torch::Tensor>> stovec,
                 std::shared_ptr<std::vector<std::string>> dup_tokens) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  // get to line we care about
  for (int64_t i = 0; i < start_line; i++) {
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
  double_conversion::StringToDoubleConverter converter(
      converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);

  for (int64_t i = start_line; i < end_line; i++) {
    std::string token;
    // read the token
    std::getline(fin, token, static_cast<char>(delimiter_ascii));

    std::string vec_val;
    std::vector<float> vec_float(vector_dim);
    // read the vector
    for (int64_t i = 0; i < vector_dim; i++) {
      fin >> vec_val;
      const char *tmp_str = vec_val.c_str();
      int processed_characters_count;
      vec_float[i] = converter.StringToFloat(tmp_str, strlen(tmp_str),
                                             &processed_characters_count);
      // TODO: Check character count?
    }
    // Shed trailing whitespace to consume next line
    fin >> std::ws;

    TORCH_CHECK(vector_dim == static_cast<int64_t>(vec_float.size()),
                "Vector for token " + token + " has " +
                    std::to_string(vec_float.size()) +
                    " but previously read vectors have " +
                    std::to_string(vector_dim) +
                    " dimensions. All vectors must have "
                    "the same number of dimensions.");

    if (stovec->contains(token)) {
      dup_tokens->push_back(token);
    } else {
      stovec->insert(std::move(token), torch::tensor(vec_float));
    }
  }
}

void concat_vectors(std::vector<std::shared_ptr<VectorsDict>> chunk_tokens,
                    std::vector<std::shared_ptr<StringList>> chunk_dup_tokens,
                    const int64_t num_lines, const int64_t vector_dim) {
  // TODO: Improve error message.
  TORCH_CHECK(chunk_tokens.size() > 0, "Must be at least 1 chunk!");
  TORCH_CHECK(chunk_tokens.size() == chunk_dup_tokens.size(),
              "tokens and dup_tokens vectors size must match!");
  auto &tokens = *chunk_tokens[0];
  auto &dup_tokens = *chunk_dup_tokens[0];

  // concat all loaded tuples
  for (size_t i = 1; i < chunk_tokens.size(); i++) {
    auto &subset_tokens = *chunk_tokens[i];
    auto &subset_dup_tokens = *chunk_dup_tokens[i];

    // efficient dup tokens concatenation
    std::move(subset_dup_tokens.begin(), subset_dup_tokens.end(),
              std::back_inserter(dup_tokens));

    for (const auto &element : subset_tokens) {
      if (tokens.contains(element.key())) {
        // TODO: This can yield duplicates within the duplicates
        // Do we want a set instead?
        dup_tokens.push_back(element.key());
      }
      tokens.insert(element.key(), element.value());
    }
  }
}

constexpr int64_t GRAIN_SIZE = 32768;
std::tuple<c10::intrusive_ptr<Vectors>, std::vector<std::string>>
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const int64_t delimiter_ascii,
                                  int64_t num_cpus,
                                  c10::optional<torch::Tensor> opt_unk_tensor) {
  std::cerr << "[INFO] Reading file " << file_path << std::endl;

  int64_t num_lines, num_header_lines, vector_dim;
  std::tie(num_lines, num_header_lines, vector_dim) =
      _infer_shape(file_path, delimiter_ascii);

  int64_t chunk_size = divup(num_lines, num_cpus);
  // Launching a thread on less lines than this likely has too much overhead.
  chunk_size = std::max(chunk_size, GRAIN_SIZE);

  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<VectorsDict>> chunk_tokens;
  std::vector<std::shared_ptr<StringList>> chunk_dup_tokens;

  // create threads
  for (int64_t i = num_header_lines; i < num_lines; i += chunk_size) {
    auto tokens_ptr = std::make_shared<VectorsDict>();
    auto dup_tokens_ptr = std::make_shared<StringList>();
    threads.push_back(std::thread(
        parse_chunk, file_path, i, std::min(num_lines, i + chunk_size),
        vector_dim, delimiter_ascii, tokens_ptr, dup_tokens_ptr));
    chunk_tokens.push_back(tokens_ptr);
    chunk_dup_tokens.push_back(dup_tokens_ptr);
  }

  // join threads
  for (auto &thread : threads) {
    thread.join();
  }

  concat_vectors(chunk_tokens, chunk_dup_tokens, num_lines, vector_dim);

  torch::Tensor unk_tensor;
  if (opt_unk_tensor) {
    unk_tensor = *opt_unk_tensor;
  } else {
    unk_tensor = torch::zeros({vector_dim});
  }
  return std::make_tuple(
      c10::make_intrusive<Vectors>(Vectors(*chunk_tokens[0], unk_tensor)),
      *chunk_dup_tokens[0]);
  // return out_tuple;
}

// Registers our custom class with torch.
static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, torch::Tensor,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("lookup_vectors", &Vectors::lookup_vectors)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<Vectors> &self)
                -> std::tuple<std::vector<std::string>, torch::Tensor,
                              torch::Tensor> {
              std::vector<std::string> tokens;
              std::vector<at::Tensor> vectors;
              for (const auto &element : self->stovec_) {
                tokens.push_back(element.key());
                vectors.push_back(element.value());
              }
              std::tuple<std::vector<std::string>, torch::Tensor, torch::Tensor>
                  states(tokens, at::stack(at::TensorList(vectors)),
                         self->unk_tensor_);
              return states;
            },
            // __getstate__
            [](std::tuple<std::vector<std::string>, torch::Tensor,
                          torch::Tensor>
                   states) -> c10::intrusive_ptr<Vectors> {
              return c10::make_intrusive<Vectors>(
                  std::move(std::get<0>(states)),
                  std::move(std::get<1>(states)),
                  std::move(std::get<2>(states)));
            });

// Registers our custom op with torch.
TORCH_LIBRARY(torchtext, m) {
  m.def("_load_token_and_vectors_from_file",
        &_load_token_and_vectors_from_file);
}

} // namespace
} // namespace torchtext
