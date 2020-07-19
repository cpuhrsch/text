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

typedef std::tuple<Dict<std::string, torch::Tensor>, std::vector<std::string>>
    LoadedVectorsTuple;

typedef std::tuple<Dict<std::string, torch::Tensor>, std::vector<std::string>>
    LoadedChunkVectorsTuple;

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

void _load_tokens_from_file_chunk(
    const std::string &file_path, const int64_t start_line,
    const int64_t num_lines, const int64_t vector_dim,
    const int64_t delimiter_ascii,
    std::promise<LoadedChunkVectorsTuple> &&promise) {
  std::ifstream fin;
  fin.open(file_path, std::ios::in);

  // TODO: Instead of using Dict, could just use Vectors class
  Dict<std::string, torch::Tensor> stovec;
  std::vector<std::string> dup_tokens;

  int64_t num_vecs_loaded = 0;

  // get to line we care about
  for (int64_t i = 0; i < start_line; i++) {
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
  double_conversion::StringToDoubleConverter converter(
      converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);

  for (int64_t i = start_line; i < start_line + num_lines; i++) {
    std::string token, vec_val, line;
    std::vector<float> vec_float;

    std::getline(fin, line);
    std::istringstream sstrm(std::move(line));

    // read the token
    std::getline(sstrm, token, static_cast<char>(delimiter_ascii));

    // read the vector
    for (int64_t i = 0; i < vector_dim; i++) {
      sstrm >> vec_val;
      const char *tmp_str = vec_val.c_str();
      int processed_characters_count;
      vec_float.push_back(converter.StringToFloat(tmp_str, strlen(tmp_str),
                                                  &processed_characters_count));
      // TODO: Check character count?
    }

    TORCH_CHECK(vector_dim == static_cast<int64_t>(vec_float.size()),
                "Vector for token " + token + " has " +
                    std::to_string(vec_float.size()) +
                    " but previously read vectors have " +
                    std::to_string(vector_dim) +
                    " dimensions. All vectors must have "
                    "the same number of dimensions.");

    if (stovec.find(token) != stovec.end()) {
      dup_tokens.push_back(token);
      continue;
    }
    stovec.insert(std::move(token), torch::tensor(vec_float));
    num_vecs_loaded++;
  }
  promise.set_value(std::make_tuple(stovec, dup_tokens));
}

void _concat_loaded_vectors_tuples(std::vector<LoadedChunkVectorsTuple> &tuples,
                                   const int64_t num_lines,
                                   const int64_t vector_dim,
                                   LoadedVectorsTuple *out_tuple) {
  // TODO: Improve error message.
  TORCH_CHECK(tuples.size() > 0, "Must be at least 1 chunk!");
  auto &&tokens = std::move(std::get<0>(tuples[0]));
  auto &&dup_tokens = std::move(std::get<1>(tuples[0]));

  // concat all loaded tuples
  for (size_t i = 1; i < tuples.size(); i++) {
    auto &&subset_tokens = std::move(std::get<0>(tuples[i]));
    auto &&subset_dup_tokens = std::move(std::get<1>(tuples[i]));
    int64_t num_subset_vecs_loaded = 0;

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
      num_subset_vecs_loaded++;
    }
  }
  // construct the vectors tensor
  *out_tuple = std::make_tuple(std::move(tokens), std::move(dup_tokens));
}

std::tuple<c10::intrusive_ptr<Vectors>, std::vector<std::string>>
_load_token_and_vectors_from_file(const std::string &file_path,
                                  const int64_t delimiter_ascii,
                                  int64_t num_cpus,
                                  c10::optional<torch::Tensor> opt_unk_tensor) {
  std::cerr << "[INFO] Reading file " << file_path << std::endl;

  std::tuple<int64_t, int64_t, int64_t> num_lines_headers_vector_dim_tuple =
      _infer_shape(file_path, delimiter_ascii);

  int64_t num_lines = std::get<0>(num_lines_headers_vector_dim_tuple);
  int64_t num_header_lines = std::get<1>(num_lines_headers_vector_dim_tuple);
  int64_t vector_dim = std::get<2>(num_lines_headers_vector_dim_tuple);

  // guard against num_lines being smaller than num_cpus
  num_cpus = std::min(num_lines, num_cpus);
  // need chunk size large enough to read entire file
  int64_t chunk_size = num_lines / num_cpus + 1;

  std::vector<std::future<LoadedChunkVectorsTuple>> futures;
  std::vector<std::thread> threads;
  std::vector<LoadedChunkVectorsTuple> tuples;

  // create threads
  for (int64_t i = 0; i < num_cpus; i++) {
    std::promise<LoadedChunkVectorsTuple> p;
    std::future<LoadedChunkVectorsTuple> f = p.get_future();
    futures.push_back(std::move(f));

    // for first chunk of file we should start from the line after all the
    // header lines
    if (i == 0 && num_header_lines > 0) {
      threads.push_back(std::thread(
          _load_tokens_from_file_chunk, file_path, num_header_lines,
          std::min(chunk_size - num_header_lines, num_lines - num_header_lines),
          vector_dim, delimiter_ascii, std::move(p)));
    } else {
      threads.push_back(
          std::thread(_load_tokens_from_file_chunk, file_path, i * chunk_size,
                      std::min(chunk_size, num_lines - (i * chunk_size)),
                      vector_dim, delimiter_ascii, std::move(p)));
    }
  }

  // join threads
  for (int64_t i = 0; i < num_cpus; i++) {
    threads[i].join();
  }

  // get all loaded tuples
  for (int64_t i = 0; i < num_cpus; i++) {
    tuples.push_back(std::move(futures[i].get()));
  }

  LoadedVectorsTuple out_tuple;
  _concat_loaded_vectors_tuples(tuples, num_lines, vector_dim, &out_tuple);

  auto tmp_tokens = std::get<0>(out_tuple);
  auto dup_tokens = std::get<1>(out_tuple);
  torch::Tensor unk_tensor;
  if (opt_unk_tensor) {
    unk_tensor = *opt_unk_tensor;
  } else {
    unk_tensor = torch::zeros({vector_dim});
  }
  return std::make_tuple(
      c10::make_intrusive<Vectors>(Vectors(tmp_tokens, unk_tensor)),
      dup_tokens);
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
