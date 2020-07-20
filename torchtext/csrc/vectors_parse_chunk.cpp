#include <climits>
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vectors_parse_chunk.h>

namespace torchtext {

typedef std::vector<std::string> StringList;

constexpr size_t SYS_BUFSIZE_MAX = INT_MAX >> 20 << 2;
constexpr size_t MAX_TOKEN_SIZE = 512;
#ifdef EINTR
#define IS_EINTR(x) ((x) == EINTR)
#else
#define IS_EINTR(x) 0
#endif

/* Read(write) up to COUNT bytes at BUF from(to) descriptor FD, retrying if
   interrupted.  Return the actual number of bytes read(written), zero for
   EOF, or SAFE_READ_ERROR(SAFE_WRITE_ERROR) upon error.  */
size_t safe_read(int fd, void *buf, size_t count) {
  for (;;) {
    ssize_t result = read(fd, buf, count);

    if (0 <= result)
      return result;
    else if (IS_EINTR(errno))
      continue;
    else if (errno == EINVAL && SYS_BUFSIZE_MAX < count)
      count = SYS_BUFSIZE_MAX;
    else
      return result;
  }
}

#define O_RDONLY 00
#define O_WRONLY 01
#define O_RDWR 02

void parse_chunk(const std::string &file_path, const int64_t start_line,
                 const int64_t end_line, const int64_t vector_dim,
                 const int64_t delimiter_ascii,
                 std::shared_ptr<StringList> tokens, float *data_ptr) {
  // std::cout << "asdfSasdfkljF" << std::endl;
  int fd = open(file_path.c_str(), O_RDONLY);
  char buf[(16 * 1024) + 1];
  char token[MAX_TOKEN_SIZE];
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

  int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
  double_conversion::StringToDoubleConverter converter(
      converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);

  int64_t line_count = 0;
  int64_t vector_pointer = 0;
  size_t token_pointer = 0;
  size_t bytes_read;
  bool read_word = true;
  // std::cout << "asdfSHDJF" << std::endl;
  while ((bytes_read = safe_read(fd, buf, (16 * 1024))) > 0) {
    char *p = buf;
    char *end = p + bytes_read;
    while (p != end && line_count < start_line) {
      line_count += *p++ == '\n';
      // std::cout << "line_count: " << line_count << std::endl;
    }
    if (p == end) {
      continue;
    }
    if (line_count < start_line) {
      continue;
    }
    while (p != end && line_count < end_line) {
      // std::cout << "SHDJF" << std::endl;
      TORCH_CHECK(token_pointer < MAX_TOKEN_SIZE,
                  "Exceeded maximum token length.");
      if (*p == static_cast<char>(delimiter_ascii) || *p == ' ') {
        if (read_word) {
          auto result_token = std::string(token, token_pointer);
          // std::cout << "found delimiter - result_token: " << result_token
          //           << " - token_pointer: " << token_pointer << std::endl;
          tokens->push_back(result_token);
          vector_pointer = 0;
          token_pointer = 0;
          p++;
          read_word = false;
          continue;
        }
        // std::cout << "found whitespace" << std::endl;
        if (token_pointer > 0) {
          // std::cout << "processing token" << std::endl;
          int processed_characters_count;
          data_ptr[line_count * vector_dim + vector_pointer] =
              converter.StringToFloat(token, token_pointer,
                                      &processed_characters_count);
          // std::cout << "processed " << processed_characters_count << std::endl;
          // if (processed_characters_count == 0) {
          //   token_pointer = 0;
          //   continue;
          // }
          TORCH_CHECK(processed_characters_count == token_pointer,
                      "Parsed less than token length: ",
                      processed_characters_count, " vs ", token_pointer);
          token_pointer = 0;
          vector_pointer++;
        }
        p++;
        continue;
      }
      if (*p == '\n') {
        // std::cout << "found newline - line_count: " << line_count << std::endl;
        line_count++;
        vector_pointer = 0;
        token_pointer = 0;
        read_word = true;
        p++;
        continue;
      }
      // std::cout << "token_pointer: " << token_pointer << std::endl;
      token[token_pointer] = *p++;
      token_pointer++;
    }
  }
}

} // namespace torchtext
