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

struct StringList {
  StringList(const std::vector<std::string>& strs) 
  {
    num_strings = strs.size();
    strlens = static_cast<size_t *>(malloc(sizeof(size_t) * num_strings));
    size_t length = 0;
    for (size_t i = 0; i < num_strings; i++) {
      const char *c_str = strs[i].c_str();
      size_t s_len = strlen(c_str);
      strlens[i] = s_len;
      length += s_len;
    }
    size_t index = 0;
    data = static_cast<char *>(malloc(sizeof(char) * length));
    for (size_t i = 0; i < num_strings; i++) {
      const char *c_str = strs[i].c_str();
      memcpy(data + index, c_str, strlens[i]);
      index += strlens[i];
    }
  }

  std::string __str__() {
    std::stringstream ss;
    size_t length = 0;
    for (size_t i = 0; i < num_strings; i++) {
      length += strlens[i];
    }
    for (size_t i = 0; i < length; i++) {
      ss << data[i];
    }
    ss << " - ";
    ss << std::vector<int64_t>(strlens, strlens + num_strings);
    return ss.str();
  }

  char* data;
  size_t* strlens;
  size_t num_strings;

};

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
  at::Tensor get_vecs_by_tokens(const StringList &tokens) {
    int64_t index = 0;
    at::Tensor indices = torch::empty({int64_t(tokens.num_strings)}, at::TensorOptions(torch::Dtype::Long));
    auto indices_accessor = indices.accessor<int64_t, 1>();
    char* data = tokens.data;
    size_t* strlens = tokens.strlens;
    size_t offset = 0;
    for (size_t i = 0; i < tokens.num_strings; i++) {
      auto search = _map.find(hash(data + offset, strlens[i]));
      offset += strlens[i];
      if (search != _map.end()) {
        indices_accessor[index] = search->second;
      } else {
        indices_accessor[index] = _unk_index;
      }
      index++;
    }
    return at::index_select(_vectors, 0, indices);
  }
  at::Tensor get_vecs_by_tokens(const std::vector<std::string> &tokens) {
    int64_t index = 0;
    at::Tensor indices = torch::empty({int64_t(tokens.size())}, at::TensorOptions(torch::Dtype::Long));
    auto indices_accessor = indices.accessor<int64_t, 1>();
    for (const std::string &token : tokens) {
      auto search = _map.find(hash(token));
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
    auto c1 = py::class_<StringList>(m, "StringList");
    c1.def(py::init<const std::vector<std::string> &>());
    c1.def("__str__", &StringList::__str__);

    auto c = py::class_<Vocab>(m, "Vocab");
    c.def(py::init<
            const std::vector<std::string>&, // stoi
            at::Tensor, // vectors
            at::Tensor>() // unk_vector
         );
    c.def("__getitem__", &Vocab::__getitem__);
    c.def("__len__", &Vocab::__len__);
    c.def("get_vecs_by_tokens",
          py::overload_cast<const std::vector<std::string> &>(
              &Vocab::get_vecs_by_tokens));
    c.def("get_vecs_by_tokens",
          py::overload_cast<const StringList &>(&Vocab::get_vecs_by_tokens));
}
