#include <stdexcept>
#include <string>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/library.h>
#include <torch/script.h>

#include <pybind11/pybind11.h>

using namespace torch::jit;

namespace py = pybind11;

std::vector<py::object> map(py::object fn, std::vector<py::object> input_list) {
  if (!py::isinstance<StrongFunctionPtr>(fn)) {
    throw std::runtime_error("Given object is not a JIT function.");
  }
  auto sfn = py::cast<StrongFunctionPtr>(fn);
  Function &operation = *sfn.function_;
  std::vector<py::object> result;
  for (size_t i = 0; i < input_list.size(); i++) {
    auto match = tryToInferType(input_list[i]);
    if (!match.success()) {
      TORCH_CHECK(false, "Argument ", std::to_string(i),
                  " cannot be converted into IValue.\n", match.reason());
    }
    auto tmp_ivalue = toIValue(input_list[i], match.type());
    auto tmp_result = operation(std::vector<c10::IValue>({tmp_ivalue}));
    result.push_back(toPyObject(tmp_result));
  }
  return result;
}

PYBIND11_MODULE(_torchtext, m) { m.def("map", &map); }
