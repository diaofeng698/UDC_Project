#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    class Logger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, char const *message) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cerr << "[TensorRT] " << message << '\n';
            }
        }
    };

    void checkCuda(cudaError_t status, char const *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    struct CudaStream
    {
        cudaStream_t value{};
        CudaStream() { checkCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
        ~CudaStream()
        {
            if (value != nullptr)
                cudaStreamDestroy(value);
        }
        CudaStream(CudaStream const &) = delete;
        CudaStream &operator=(CudaStream const &) = delete;
    };

    struct CudaEvent
    {
        cudaEvent_t value{};
        CudaEvent() { checkCuda(cudaEventCreate(&value), "cudaEventCreate"); }
        ~CudaEvent()
        {
            if (value != nullptr)
                cudaEventDestroy(value);
        }
        CudaEvent(CudaEvent const &) = delete;
        CudaEvent &operator=(CudaEvent const &) = delete;
    };

    struct Buffer
    {
        std::string name;
        nvinfer1::TensorIOMode mode{};
        nvinfer1::DataType dtype{};
        nvinfer1::Dims shape{};
        std::size_t bytes{};
        void *host{};
        void *device{};

        ~Buffer()
        {
            if (device != nullptr)
                cudaFree(device);
            if (host != nullptr)
                cudaFreeHost(host);
        }
        Buffer() = default;
        Buffer(Buffer const &) = delete;
        Buffer &operator=(Buffer const &) = delete;
        Buffer(Buffer &&other) noexcept
            : name(std::move(other.name)), mode(other.mode), dtype(other.dtype), shape(other.shape),
              bytes(other.bytes), host(other.host), device(other.device)
        {
            other.host = nullptr;
            other.device = nullptr;
        }
        Buffer &operator=(Buffer &&) = delete;
    };

    struct Options
    {
        std::string engine;
        std::string json;
        int warmup{20};
        int iterations{200};
        bool transfers{true};
        std::map<std::string, nvinfer1::Dims> inputShapes;
    };

    struct Stats
    {
        double mean{};
        double minimum{};
        double maximum{};
        double stddev{};
        double p50{};
        double p90{};
        double p95{};
        double p99{};
    };

    std::string dimsString(nvinfer1::Dims const &dims)
    {
        std::ostringstream stream;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (i != 0)
                stream << 'x';
            stream << dims.d[i];
        }
        return stream.str();
    }

    nvinfer1::Dims parseDims(std::string const &text)
    {
        nvinfer1::Dims dims{};
        std::string normalized = text;
        std::replace(normalized.begin(), normalized.end(), ',', 'x');
        std::stringstream stream(normalized);
        std::string token;
        while (std::getline(stream, token, 'x'))
        {
            if (token.empty() || dims.nbDims >= nvinfer1::Dims::MAX_DIMS)
            {
                throw std::runtime_error("invalid shape: " + text);
            }
            long value = std::stol(token);
            if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
            {
                throw std::runtime_error("shape dimensions must be positive int32 values: " + text);
            }
            dims.d[dims.nbDims++] = static_cast<std::int32_t>(value);
        }
        if (dims.nbDims == 0)
            throw std::runtime_error("empty shape: " + text);
        return dims;
    }

    std::size_t elementSize(nvinfer1::DataType dtype)
    {
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT4:
            return 1;
#endif
        default:
            throw std::runtime_error("unsupported TensorRT I/O data type");
        }
    }

    std::string dtypeString(nvinfer1::DataType dtype)
    {
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8:
            return "UINT8";
        case nvinfer1::DataType::kFP8:
            return "FP8";
        case nvinfer1::DataType::kBF16:
            return "BF16";
        case nvinfer1::DataType::kINT64:
            return "INT64";
        case nvinfer1::DataType::kINT4:
            return "INT4";
#endif
        default:
            return "UNKNOWN";
        }
    }

    std::size_t volume(nvinfer1::Dims const &dims)
    {
        std::size_t result = 1;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] <= 0)
                throw std::runtime_error("unresolved dynamic shape: " + dimsString(dims));
            auto value = static_cast<std::size_t>(dims.d[i]);
            if (result > std::numeric_limits<std::size_t>::max() / value)
                throw std::overflow_error("tensor volume overflow");
            result *= value;
        }
        return result;
    }

    std::vector<char> readBinary(std::string const &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("cannot open engine: " + path);
        auto size = file.tellg();
        if (size <= 0)
            throw std::runtime_error("engine is empty: " + path);
        std::vector<char> data(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(data.data(), size))
            throw std::runtime_error("cannot read engine: " + path);
        return data;
    }

    double percentile(std::vector<double> sorted, double fraction)
    {
        std::sort(sorted.begin(), sorted.end());
        double position = fraction * static_cast<double>(sorted.size() - 1);
        auto low = static_cast<std::size_t>(std::floor(position));
        auto high = static_cast<std::size_t>(std::ceil(position));
        double weight = position - static_cast<double>(low);
        return sorted[low] * (1.0 - weight) + sorted[high] * weight;
    }

    Stats calculate(std::vector<double> const &values)
    {
        Stats stats{};
        stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
        stats.minimum = *minimum;
        stats.maximum = *maximum;
        double squared = 0.0;
        for (double value : values)
            squared += (value - stats.mean) * (value - stats.mean);
        stats.stddev = std::sqrt(squared / static_cast<double>(values.size()));
        stats.p50 = percentile(values, 0.50);
        stats.p90 = percentile(values, 0.90);
        stats.p95 = percentile(values, 0.95);
        stats.p99 = percentile(values, 0.99);
        return stats;
    }

    void printStats(std::string const &name, Stats const &stats)
    {
        std::cout << std::left << std::setw(12) << name << std::right
                  << " mean=" << std::setw(9) << std::fixed << std::setprecision(4) << stats.mean
                  << " p50=" << std::setw(9) << stats.p50
                  << " p90=" << std::setw(9) << stats.p90
                  << " p95=" << std::setw(9) << stats.p95
                  << " p99=" << std::setw(9) << stats.p99
                  << " min=" << std::setw(9) << stats.minimum
                  << " max=" << std::setw(9) << stats.maximum
                  << " std=" << std::setw(9) << stats.stddev << " ms\n";
    }

    std::string jsonEscape(std::string const &value)
    {
        std::ostringstream stream;
        for (char character : value)
        {
            if (character == '"' || character == '\\')
                stream << '\\';
            stream << character;
        }
        return stream.str();
    }

    std::uint64_t fnv1a(void const *data, std::size_t size)
    {
        auto const *bytes = static_cast<unsigned char const *>(data);
        std::uint64_t hash = 1469598103934665603ULL;
        for (std::size_t index = 0; index < size; ++index)
        {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    void writeStatsJson(std::ostream &output, std::string const &name, Stats const &stats, bool comma)
    {
        output << "    \"" << name << "\": {\"mean\": " << stats.mean
               << ", \"p50\": " << stats.p50 << ", \"p90\": " << stats.p90
               << ", \"p95\": " << stats.p95 << ", \"p99\": " << stats.p99
               << ", \"min\": " << stats.minimum << ", \"max\": " << stats.maximum
               << ", \"stddev\": " << stats.stddev << "}" << (comma ? "," : "") << '\n';
    }

    Options parseOptions(int argc, char **argv)
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            std::string argument = argv[index];
            auto requireValue = [&](char const *option) -> std::string
            {
                if (++index >= argc)
                    throw std::runtime_error(std::string("missing value for ") + option);
                return argv[index];
            };
            if (argument == "--engine")
                options.engine = requireValue("--engine");
            else if (argument == "--warmup")
                options.warmup = std::stoi(requireValue("--warmup"));
            else if (argument == "--iterations")
                options.iterations = std::stoi(requireValue("--iterations"));
            else if (argument == "--json")
                options.json = requireValue("--json");
            else if (argument == "--no-transfers")
                options.transfers = false;
            else if (argument == "--shape")
            {
                std::string specification = requireValue("--shape");
                auto equals = specification.find('=');
                if (equals == std::string::npos)
                    throw std::runtime_error("--shape format is tensor=1x3xHxW");
                options.inputShapes[specification.substr(0, equals)] = parseDims(specification.substr(equals + 1));
            }
            else if (argument == "--help")
            {
                std::cout << "Usage: bnudc_trt_infer --engine model.plan [--warmup 20] [--iterations 200]\n"
                             "       [--shape input=1x3x512x416] [--no-transfers] [--json result.json]\n";
                std::exit(0);
            }
            else
                throw std::runtime_error("unknown argument: " + argument);
        }
        if (options.engine.empty())
            throw std::runtime_error("--engine is required");
        if (options.warmup < 0 || options.iterations <= 0)
            throw std::runtime_error("warmup must be >=0 and iterations must be >0");
        return options;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        Options options = parseOptions(argc, argv);
        Logger logger;
        initLibNvInferPlugins(&logger, "");

        int device = 0;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDevice(&device), "cudaGetDevice");
        checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

        auto serialized = readBinary(options.engine);
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime)
            throw std::runtime_error("createInferRuntime failed");
        std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(serialized.data(), serialized.size())};
        if (!engine)
            throw std::runtime_error("deserializeCudaEngine failed; verify TensorRT/GPU compatibility");
        std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
        if (!context)
            throw std::runtime_error("createExecutionContext failed");

        for (auto const &[name, shape] : options.inputShapes)
        {
            if (engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
            {
                throw std::runtime_error("--shape tensor is not an input: " + name);
            }
            if (!context->setInputShape(name.c_str(), shape))
                throw std::runtime_error("setInputShape failed for " + name);
        }
        if (!context->allInputDimensionsSpecified())
            throw std::runtime_error("dynamic input dimensions unresolved; provide --shape name=...");

        std::vector<Buffer> buffers;
        std::size_t inputBytes = 0;
        std::size_t outputBytes = 0;
        int inputCount = 0;
        int outputCount = 0;
        for (int index = 0; index < engine->getNbIOTensors(); ++index)
        {
            char const *tensorName = engine->getIOTensorName(index);
            if (tensorName == nullptr)
                throw std::runtime_error("getIOTensorName returned null");
            Buffer buffer;
            buffer.name = tensorName;
            buffer.mode = engine->getTensorIOMode(tensorName);
            buffer.dtype = engine->getTensorDataType(tensorName);
            buffer.shape = context->getTensorShape(tensorName);
            buffer.bytes = volume(buffer.shape) * elementSize(buffer.dtype);
            checkCuda(cudaMallocHost(&buffer.host, buffer.bytes), "cudaMallocHost");
            checkCuda(cudaMalloc(&buffer.device, buffer.bytes), "cudaMalloc");
            std::memset(buffer.host, 0, buffer.bytes);
            if (!context->setTensorAddress(tensorName, buffer.device))
                throw std::runtime_error("setTensorAddress failed for " + buffer.name);
            if (buffer.mode == nvinfer1::TensorIOMode::kINPUT)
            {
                inputBytes += buffer.bytes;
                ++inputCount;
            }
            else
            {
                outputBytes += buffer.bytes;
                ++outputCount;
            }
            buffers.emplace_back(std::move(buffer));
        }

        std::cout << "TensorRT C++ inference\n"
                  << "  TensorRT compile version: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '\n'
                  << "  GPU: " << properties.name << " (SM " << properties.major << '.' << properties.minor << ")\n"
                  << "  Engine: " << options.engine << '\n'
                  << "  Warmup/iterations: " << options.warmup << '/' << options.iterations << '\n'
                  << "  Transfers: " << (options.transfers ? "H2D + inference + D2H" : "inference only") << '\n';
        for (Buffer const &buffer : buffers)
        {
            std::cout << "  " << (buffer.mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " : "OUTPUT")
                      << ' ' << buffer.name << " shape=" << dimsString(buffer.shape)
                      << " dtype=" << dtypeString(buffer.dtype) << " bytes=" << buffer.bytes << '\n';
        }

        CudaStream stream;
        CudaEvent h2dStart, h2dEnd, inferStart, inferEnd, d2hStart, d2hEnd;
        auto execute = [&](bool measure, std::vector<double> *h2d, std::vector<double> *inference,
                           std::vector<double> *d2h, std::vector<double> *endToEnd)
        {
            auto cpuStart = std::chrono::steady_clock::now();
            checkCuda(cudaEventRecord(h2dStart.value, stream.value), "record H2D start");
            if (options.transfers)
            {
                for (Buffer const &buffer : buffers)
                    if (buffer.mode == nvinfer1::TensorIOMode::kINPUT)
                        checkCuda(cudaMemcpyAsync(buffer.device, buffer.host, buffer.bytes, cudaMemcpyHostToDevice, stream.value), "input H2D");
            }
            checkCuda(cudaEventRecord(h2dEnd.value, stream.value), "record H2D end");
            checkCuda(cudaEventRecord(inferStart.value, stream.value), "record inference start");
            if (!context->enqueueV3(stream.value))
                throw std::runtime_error("enqueueV3 failed");
            checkCuda(cudaEventRecord(inferEnd.value, stream.value), "record inference end");
            checkCuda(cudaEventRecord(d2hStart.value, stream.value), "record D2H start");
            if (options.transfers)
            {
                for (Buffer const &buffer : buffers)
                    if (buffer.mode == nvinfer1::TensorIOMode::kOUTPUT)
                        checkCuda(cudaMemcpyAsync(buffer.host, buffer.device, buffer.bytes, cudaMemcpyDeviceToHost, stream.value), "output D2H");
            }
            checkCuda(cudaEventRecord(d2hEnd.value, stream.value), "record D2H end");
            checkCuda(cudaStreamSynchronize(stream.value), "cudaStreamSynchronize");
            auto cpuEnd = std::chrono::steady_clock::now();
            if (measure)
            {
                float h2dMs{}, inferMs{}, d2hMs{};
                checkCuda(cudaEventElapsedTime(&h2dMs, h2dStart.value, h2dEnd.value), "H2D elapsed");
                checkCuda(cudaEventElapsedTime(&inferMs, inferStart.value, inferEnd.value), "inference elapsed");
                checkCuda(cudaEventElapsedTime(&d2hMs, d2hStart.value, d2hEnd.value), "D2H elapsed");
                h2d->push_back(h2dMs);
                inference->push_back(inferMs);
                d2h->push_back(d2hMs);
                endToEnd->push_back(std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count());
            }
        };

        for (int index = 0; index < options.warmup; ++index)
            execute(false, nullptr, nullptr, nullptr, nullptr);
        std::vector<double> h2d, inference, d2h, endToEnd;
        h2d.reserve(options.iterations);
        inference.reserve(options.iterations);
        d2h.reserve(options.iterations);
        endToEnd.reserve(options.iterations);
        for (int index = 0; index < options.iterations; ++index)
            execute(true, &h2d, &inference, &d2h, &endToEnd);

        Stats h2dStats = calculate(h2d);
        Stats inferenceStats = calculate(inference);
        Stats d2hStats = calculate(d2h);
        Stats endToEndStats = calculate(endToEnd);
        std::cout << "\nLatency statistics\n";
        if (options.transfers)
            printStats("H2D", h2dStats);
        printStats("Inference", inferenceStats);
        if (options.transfers)
            printStats("D2H", d2hStats);
        printStats("End-to-end", endToEndStats);
        std::cout << "\nEfficiency indicators\n"
                  << "  Inference throughput: " << std::fixed << std::setprecision(2) << 1000.0 / inferenceStats.mean << " infer/s\n"
                  << "  Serial E2E throughput: " << 1000.0 / endToEndStats.mean << " infer/s\n"
                  << "  Inference jitter CV: " << inferenceStats.stddev / inferenceStats.mean * 100.0 << "%\n"
                  << "  Transfer bytes H2D/D2H: " << inputBytes << '/' << outputBytes << '\n';
        if (options.transfers)
        {
            std::cout << "\nOutput checksums (FNV-1a, dummy input)\n";
            for (Buffer const &buffer : buffers)
            {
                if (buffer.mode == nvinfer1::TensorIOMode::kOUTPUT)
                {
                    std::cout << "  " << buffer.name << ": 0x" << std::hex << fnv1a(buffer.host, buffer.bytes) << std::dec << '\n';
                }
            }
        }

        if (!options.json.empty())
        {
            std::ofstream output(options.json);
            if (!output)
                throw std::runtime_error("cannot write JSON: " + options.json);
            output << std::fixed << std::setprecision(8)
                   << "{\n  \"engine\": \"" << jsonEscape(options.engine) << "\",\n"
                   << "  \"gpu\": \"" << jsonEscape(properties.name) << "\",\n"
                   << "  \"tensorrt_compile_version\": \"" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << "\",\n"
                   << "  \"warmup\": " << options.warmup << ",\n  \"iterations\": " << options.iterations << ",\n"
                   << "  \"transfers\": " << (options.transfers ? "true" : "false") << ",\n"
                   << "  \"input_bytes\": " << inputBytes << ",\n  \"output_bytes\": " << outputBytes << ",\n"
                   << "  \"input_count\": " << inputCount << ",\n  \"output_count\": " << outputCount << ",\n"
                   << "  \"output_checksums_fnv1a\": {";
            bool firstChecksum = true;
            if (options.transfers)
            {
                for (Buffer const &buffer : buffers)
                {
                    if (buffer.mode != nvinfer1::TensorIOMode::kOUTPUT)
                        continue;
                    output << (firstChecksum ? "\n" : ",\n") << "    \"" << jsonEscape(buffer.name) << "\": \"0x"
                           << std::hex << fnv1a(buffer.host, buffer.bytes) << std::dec << "\"";
                    firstChecksum = false;
                }
            }
            if (!firstChecksum)
                output << '\n';
            output << "  },\n"
                   << "  \"latency_ms\": {\n";
            writeStatsJson(output, "h2d", h2dStats, true);
            writeStatsJson(output, "inference", inferenceStats, true);
            writeStatsJson(output, "d2h", d2hStats, true);
            writeStatsJson(output, "end_to_end", endToEndStats, false);
            output << "  },\n  \"throughput_infer_per_sec\": " << 1000.0 / inferenceStats.mean
                   << ",\n  \"throughput_e2e_per_sec\": " << 1000.0 / endToEndStats.mean
                   << ",\n  \"inference_jitter_cv_percent\": " << inferenceStats.stddev / inferenceStats.mean * 100.0 << "\n}\n";
            std::cout << "  JSON report: " << options.json << '\n';
        }
        return 0;
    }
    catch (std::exception const &error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}