#include <torch/extension.h>
#include <unordered_map>
#include <c10/util/SmallVector.h>

// Fowler-Noll-Vo hash function
uint32_t hash(const char *c_str, size_t len) {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < len; i++) {
    h = h ^ uint32_t(static_cast<int8_t>(c_str[i]));
    h = h * 16777619;
  }
  return h;
}

uint32_t hash(const std::string &str) {
  return hash(str.c_str(), strlen(str.c_str()));
}

// Fowler-Noll-Vo hash function
void hashes(const std::vector<std::string> &strs, 
    uint32_t* h_vec, size_t len) {
  for (size_t i = 0; i < len; i++) {
    h_vec[i] = 2166136261;
  }
  for (size_t i = 0; i < len; i++) {
    const char* c_str = strs[i].c_str();
    for (size_t j = 0; j < strlen(c_str); j++) {
      h_vec[i] = h_vec[i] ^ uint32_t(static_cast<int8_t>(c_str[j]));
      h_vec[i] = h_vec[i] * 16777619;
    }
  }
}

struct Vocab {
  Vocab(std::vector<std::string> itos, at::Tensor vectors,
        at::Tensor unk_vector)
      : _vectors(
            at::cat({vectors, unk_vector.reshape({1, unk_vector.size(0)})})),
        _unk_index(vectors.size(0)) {
    int64_t index = 0;
    _map.reserve(itos.size());
    for (const std::string & t : itos) {
      _map.insert({hash(t), index});
      index++;
    }
  }
  at::Tensor __getitem__(const std::string &token) {
    auto search = _map.find(hash(token));
    if (search == _map.end()) {
      return _vectors[_vectors.size(0) - 1].clone();
    }
    return _vectors[search->second];
  }
  int64_t __len__() { return _unk_index; }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    int64_t index = 0;
    at::Tensor indices = torch::empty({int64_t(tokens.size())},
                                      at::TensorOptions(torch::Dtype::Long));
    auto indices_accessor = indices.accessor<int64_t, 1>();
    size_t len = tokens.size();
    uint32_t *h_vec = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * len));
    hashes(tokens, h_vec, len);
    for (size_t i = 0; i < len; i++) {
      auto search = _map.find(h_vec[i]);
      if (search != _map.end()) {
        indices_accessor[index] = search->second;
      } else {
        indices_accessor[index] = _unk_index;
      }
      index++;
    }
    return at::index_select(_vectors, 0, indices);
  }

private:
  std::unordered_map<uint32_t, int64_t> _map;
  at::Tensor _vectors;
  int64_t _unk_index;
};

PYBIND11_MODULE(_torchtext, m) { 
    auto c = py::class_<Vocab>(m, "Vocab");
    c.def(py::init<
            const std::vector<std::string>&, // stoi
            at::Tensor, // vectors
            at::Tensor>() // unk_vector
         );
    c.def("__getitem__", &Vocab::__getitem__);
    c.def("__len__", &Vocab::__len__);
    c.def("get_vecs_by_tokens", &Vocab::get_vecs_by_tokens);
}
