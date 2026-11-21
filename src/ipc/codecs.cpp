#include "ipc/codecs.hpp"

#include <inttypes.h>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "nanoarrow_errors.hpp"
#include "zstd.h"

namespace duckdb {
namespace ext_nanoarrow {

// A nanoarrow built without its codecs (NANOARROW_IPC_COMPRESSION=OFF) has no zstd, so
// record batch bodies are compressed and decompressed with DuckDB's bundled zstd instead.
// Dictionary batches are decoded by a decoder nanoarrow creates internally, which keeps
// nanoarrow's default decompressor until apache/arrow-nanoarrow#994 lands.
static bool UseDuckDBZstd(ArrowIpcCompressionType type) {
  return type == NANOARROW_IPC_COMPRESSION_TYPE_ZSTD &&
         ArrowIpcGetZstdDecompressionFunction() == nullptr;
}

// As nanoarrow's own zstd codec (ipc/codecs.c), with DuckDB's zstd
static ArrowErrorCode DuckDBZstdDecompress(ArrowBufferView src, uint8_t* dst,
                                           int64_t dst_size, ArrowError* error) {
  auto code =
      duckdb_zstd::ZSTD_decompress(dst, static_cast<size_t>(dst_size), src.data.data,
                                   static_cast<size_t>(src.size_bytes));
  if (duckdb_zstd::ZSTD_isError(code)) {
    ArrowErrorSet(error,
                  "ZSTD_decompress([buffer with %" PRId64
                  " bytes] -> [buffer with %" PRId64 " bytes]) failed with error '%s'",
                  src.size_bytes, dst_size, duckdb_zstd::ZSTD_getErrorName(code));
    return EIO;
  }
  if (static_cast<int64_t>(code) != dst_size) {
    ArrowErrorSet(error,
                  "Expected decompressed size of %" PRId64 " bytes but got %" PRId64
                  " bytes",
                  dst_size, static_cast<int64_t>(code));
    return EIO;
  }
  return NANOARROW_OK;
}

static ArrowErrorCode DuckDBZstdCompress(ArrowBufferView src, int compression_level,
                                         ArrowBuffer* dst, ArrowError* error) {
  auto capacity = duckdb_zstd::ZSTD_compressBound(static_cast<size_t>(src.size_bytes));
  if (duckdb_zstd::ZSTD_isError(capacity)) {
    ArrowErrorSet(error, "ZSTD_compressBound(%" PRId64 ") failed with error '%s'",
                  src.size_bytes, duckdb_zstd::ZSTD_getErrorName(capacity));
    return EINVAL;
  }
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferReserve(dst, static_cast<int64_t>(capacity)), error);
  auto code =
      duckdb_zstd::ZSTD_compress(dst->data + dst->size_bytes, capacity, src.data.data,
                                 static_cast<size_t>(src.size_bytes), compression_level);
  if (duckdb_zstd::ZSTD_isError(code)) {
    ArrowErrorSet(error,
                  "ZSTD_compress([buffer with %" PRId64 " bytes]) failed with error '%s'",
                  src.size_bytes, duckdb_zstd::ZSTD_getErrorName(code));
    return EIO;
  }
  dst->size_bytes += static_cast<int64_t>(code);
  return NANOARROW_OK;
}

static ArrowIpcCompressionType ParseArrowIpcCompressionType(const string& name) {
  // Accept the spellings DuckDB users know from other COPY formats on top of nanoarrow's
  auto lname = StringUtil::Lower(name);
  if (lname == "uncompressed") {
    lname = "none";
  } else if (lname == "lz4_frame") {
    lname = "lz4";
  }

  ArrowIpcCompressionType type;
  ArrowError error{};
  if (ArrowIpcCompressionTypeFromString(lname.c_str(), &type, &error) != NANOARROW_OK) {
    throw BinderException(
        "Unsupported compression type \"%s\" for Arrow IPC, expected one of "
        "'uncompressed', 'zstd' or 'lz4'",
        name);
  }
  return type;
}

bool ArrowIpcCompressionOptions::TrySetOption(const string& name, const Value& value) {
  if (StringUtil::CIEquals(name, "compression")) {
    type = ParseArrowIpcCompressionType(value.ToString());
  } else if (StringUtil::CIEquals(name, "compression_level")) {
    level = value.GetValue<int64_t>();
    level_set = true;
  } else {
    return false;
  }
  return true;
}

void ArrowIpcCompressionOptions::Validate() const {
  if (type == NANOARROW_IPC_COMPRESSION_TYPE_NONE) {
    if (level_set) {
      throw BinderException(
          "COMPRESSION_LEVEL requires COMPRESSION to be set to 'zstd' or 'lz4'");
    }
    return;
  }

  // Checked at bind time, before any encoder exists. The range also tells whether this
  // build has the codec at all: lz4 needs nanoarrow's codecs
  int min_level;
  int max_level;
  if (UseDuckDBZstd(type)) {
    min_level = duckdb_zstd::ZSTD_minCLevel();
    max_level = duckdb_zstd::ZSTD_maxCLevel();
  } else if (ArrowIpcGetCompressionLevelRange(type, &min_level, &max_level) !=
             NANOARROW_OK) {
    throw NotImplementedException(
        "Arrow IPC compression with %s is not available in this build",
        ArrowIpcCompressionTypeToString(type));
  }
  if (level_set && (level < min_level || level > max_level)) {
    throw BinderException("Compression level for %s must be between %d and %d",
                          ArrowIpcCompressionTypeToString(type), min_level, max_level);
  }
}

nanoarrow::ipc::UniqueDecoder NewDuckDBArrowDecoder() {
  nanoarrow::ipc::UniqueDecoder decoder;
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderInit(decoder.get()));
  if (UseDuckDBZstd(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD)) {
    nanoarrow::ipc::UniqueDecompressor decompressor;
    NANOARROW_THROW_NOT_OK(ArrowIpcSerialDecompressor(decompressor.get()));
    NANOARROW_THROW_NOT_OK(ArrowIpcSerialDecompressorSetFunction(
        decompressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD, &DuckDBZstdDecompress));
    NANOARROW_THROW_NOT_OK(
        ArrowIpcDecoderSetDecompressor(decoder.get(), decompressor.get()));
  }
  // Otherwise nanoarrow adds its serial decompressor on first use
  return decoder;
}

void SetArrowIpcEncoderCompression(ArrowIpcEncoder& encoder,
                                   const ArrowIpcCompressionOptions& options) {
  if (UseDuckDBZstd(options.type)) {
    nanoarrow::ipc::UniqueCompressor compressor;
    NANOARROW_THROW_NOT_OK(ArrowIpcSerialCompressor(compressor.get(), options.type,
                                                    static_cast<int>(options.level)));
    NANOARROW_THROW_NOT_OK(ArrowIpcSerialCompressorSetFunction(
        compressor.get(), options.type, &DuckDBZstdCompress));
    NANOARROW_THROW_NOT_OK(ArrowIpcEncoderSetCompressor(&encoder, compressor.get()));
    return;
  }
  // Installs nanoarrow's serial compressor with its own codec
  ArrowError error{};
  THROW_NOT_OK(InternalException, &error,
               ArrowIpcEncoderSetCompression(&encoder, options.type,
                                             static_cast<int>(options.level), &error));
}

}  // namespace ext_nanoarrow
}  // namespace duckdb
