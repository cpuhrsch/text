// #include <climits>
// #include <cstdio>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <fcntl.h>
// #include <errno.h>
// #include <unistd.h>

// constexpr size_t SYS_BUFSIZE_MAX = INT_MAX >> 20 << 2;
// constexpr size_t MAX_TOKEN_SIZE = 1024;
// #ifdef EINTR
// #define IS_EINTR(x) ((x) == EINTR)
// #else
// #define IS_EINTR(x) 0
// #endif
//
// /* Read(write) up to COUNT bytes at BUF from(to) descriptor FD, retrying if
//    interrupted.  Return the actual number of bytes read(written), zero for
//    EOF, or SAFE_READ_ERROR(SAFE_WRITE_ERROR) upon error.  */
// size_t safe_read(int fd, void *buf, size_t count) {
//   for (;;) {
//     ssize_t result = read(fd, buf, count);
//
//     if (0 <= result)
//       return result;
//     else if (IS_EINTR(errno))
//       continue;
//     else if (errno == EINVAL && SYS_BUFSIZE_MAX < count)
//       count = SYS_BUFSIZE_MAX;
//     else
//       return result;
//   }
// }
//
// #define O_RDONLY 00
// #define O_WRONLY 01
// #define O_RDWR 02
//
// void parse_chunk(const std::string &file_path, const int64_t start_line,
//                  const int64_t end_line, const int64_t vector_dim,
//                  const int64_t delimiter_ascii,
//                  std::shared_ptr<StringList> tokens, float *data_ptr) {
//   int fd = open(file_path.c_str(), O_RDONLY);
//   char buf[(16 * 1024) + 1];
//   char token[MAX_TOKEN_SIZE];
//   size_t token_pointer = 0;
//   posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
//
//   int converter_flags = double_conversion::StringToDoubleConverter::NO_FLAGS;
//   double_conversion::StringToDoubleConverter converter(
//       converter_flags, 0.0f, double_conversion::Single::NaN(), NULL, NULL);
//
//   int64_t line_count = 0;
//   int64_t vector_pointer = 0;
//   size_t bytes_read;
//   while ((bytes_read = safe_read(fd, buf, (16 * 1024))) > 0) {
//     char *p = buf;
//     char *end = p + bytes_read;
//     while (p != end && line_count < start_line)
//       line_count += *p++ == '\n';
//     if (p == end) {
//       continue;
//     }
//     if (line_count < start_line) {
//       continue;
//     }
//     while (p != end && line_count < end_line) {
//       TORCH_CHECK(token_pointer < MAX_TOKEN_SIZE,
//                   "Exceeded maximum token length.");
//       if (*p++ == static_cast<char>(delimiter_ascii)) {
//         tokens->push_back(std::string(token, token_pointer));
//         token_pointer = 0;
//         continue;
//       }
//       if (*p++ == ' ') {
//         if (token_pointer > 0) {
//           int processed_characters_count;
//           data_ptr[line_count * vector_dim + vector_pointer] =
//               converter.StringToFloat(token, token_pointer,
//                                       &processed_characters_count);
//           if (processed_characters_count == 0) {
//             continue;
//           }
//           TORCH_CHECK(processed_characters_count == token_pointer,
//                       "Parsed less than token length: ",
//                       processed_characters_count, " vs ", token_pointer);
//           token_pointer = 0;
//         }
//       }
//       if (*p++ == '\n') {
//         line_count++;
//         vector_pointer = 0;
//         continue;
//       }
//       token[token_pointer] = *p++;
//       token_pointer++;
//     }
//   }
// }
