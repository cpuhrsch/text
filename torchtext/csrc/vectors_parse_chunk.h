#pragma once
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

namespace torchtext {

void parse_chunk(const std::string &file_path, size_t offset,
                 const int64_t start_line, const int64_t end_line,
                 const int64_t vector_dim, const int64_t delimiter_ascii,
                 std::shared_ptr<std::vector<std::string>> tokens,
                 float *data_ptr);

}
