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
constexpr size_t BUFSIZE = 64 * 1024;
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

void parse_chunk(const std::string &file_path, size_t offset,
                 const int64_t start_line, const int64_t end_line,
                 const int64_t vector_dim, const int64_t delimiter_ascii,
                 std::shared_ptr<StringList> tokens, float *data_ptr) {
  tokens->reserve(end_line - start_line);
  // TODO: This needs error checking
  int fd = open(file_path.c_str(), O_RDONLY);
  char buf[BUFSIZE + 1];
  char token[MAX_TOKEN_SIZE];
  // TODO: This surely isn't portable. Use a generic C++ implementation for
  // unsupported platforms.
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_NOREUSE);
  lseek(fd, offset, SEEK_SET);

  int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
  double_conversion::StringToDoubleConverter converter(
      converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);

  int64_t line_count = start_line;
  int64_t vector_pointer = 0;
  size_t token_pointer = 0;
  size_t bytes_read;
  bool read_word = true;
  while ((bytes_read = safe_read(fd, buf, BUFSIZE)) > 0) {
    char *p = buf;
    char *end = p + bytes_read;
    // while (p != end && line_count < start_line) {
    //   line_count += *p++ == '\n';
    // }
    // if (p == end) {
    //   continue;
    // }
    // if (line_count < start_line) {
    //   continue;
    // }
    while (p != end && line_count < end_line) {
      TORCH_CHECK(token_pointer < MAX_TOKEN_SIZE,
                  "Exceeded maximum token length.");
      if (*p == static_cast<char>(delimiter_ascii) || *p == ' ') {
        if (read_word) {
          auto result_token = std::string(token, token_pointer);
          tokens->push_back(result_token);
          vector_pointer = 0;
          token_pointer = 0;
          p++;
          read_word = false;
          continue;
        }
        if (token_pointer > 0) {
          int processed_characters_count;
          data_ptr[line_count * vector_dim + vector_pointer] =
              converter.StringToFloat(token, token_pointer,
                                      &processed_characters_count);
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
        line_count++;
        vector_pointer = 0;
        token_pointer = 0;
        read_word = true;
        p++;
        continue;
      }
      token[token_pointer] = *p++;
      token_pointer++;
    }
  }
  close(fd);
}

} // namespace torchtext
