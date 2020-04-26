#include <torch/extension.h>
#include <unordered_map>

// Fowler-Noll-Vo hash function
uint32_t hash(const std::string& str) {
  const char* c_str = str.c_str();
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(static_cast<int8_t>(c_str[i]));
    h = h * 16777619;
  }
  return h;
}

struct Vocab {
  Vocab(std::vector<std::string> itos, at::Tensor vectors,
        at::Tensor unk_vector)
      : _vectors(
            at::cat({vectors, unk_vector.reshape({1, unk_vector.size(0)})})) {
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
  // -1 because of unk_vector
  int64_t __len__() { return _vectors.size(0) - 1; }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    std::vector<int64_t> indices(tokens.size(), _vectors.size(0) - 1);
    int64_t index = 0;
    for (const std::string &token : tokens) {
      auto search = _map.find(hash(token));
      if (search != _map.end()) {
        indices[index] = search->second;
      }
      index++;
    }
    at::Tensor ind = torch::tensor(indices);
    return at::index_select(_vectors, 0, ind);
  }

private:
  std::unordered_map<uint32_t, int64_t> _map;
  at::Tensor _vectors;
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
