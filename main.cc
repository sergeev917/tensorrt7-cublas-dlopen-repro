#include <iostream>
#include <limits>
#include <memory>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using std::cerr;
using std::ifstream;
using std::ios;
using std::ios_base;
using std::numeric_limits;
using std::size_t;
using std::streamsize;
using std::unique_ptr;

using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using nvinfer1::NetworkDefinitionCreationFlag;
using nvinfer1::createInferBuilder;
using nvonnxparser::ErrorCode;
using nvonnxparser::IParser;
using nvonnxparser::createParser;

struct InferLogger: ILogger {
    void log(Severity severity, const char * msg) override {
        cerr << msg << '\n';
    }
};

int main(int argc, char * argv[])
{
    if (argc != 2) {
        cerr << "expecting one argument: onnx path\n";
        return 1;
    }
    const char * path = argv[1];
    unique_ptr<char[]> onnx_buf;
    size_t onnx_buf_size;
    ifstream f(path, ios::binary);
    if (!f) {
        cerr << "cannot open file: " << path;
        return 1;
    }
    f.exceptions(ifstream::failbit);
    try {
        f.ignore(numeric_limits<streamsize>::max());
        onnx_buf_size = f.gcount();
        f.clear(); /* clear the eof flag we hit */
        f.seekg(0, ios_base::beg);
        onnx_buf.reset(new char[onnx_buf_size]);
        f.read(onnx_buf.get(), onnx_buf_size);
    } catch (const ios_base::failure &) {
        cerr << "failed to load file: " << path;
        return 1;
    }
    InferLogger logger;
    // NOTE: leaking objects here and later, toy example
    IBuilder * builder_ptr = createInferBuilder(logger);
    if (builder_ptr == nullptr) {
        cerr << "createInferBuilder() failed\n";
        return 1;
    }
    INetworkDefinition * netdef_ptr = builder_ptr->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    );
    if (netdef_ptr == nullptr) {
        cerr << "IBuilder::createNetwork() failed\n";
        return 1;
    }
    IParser * parser_ptr = createParser(*netdef_ptr, logger);
    if (parser_ptr == nullptr) {
        cerr << "createParser() failed\n";
        return 1;
    }
    if (!parser_ptr->parse(onnx_buf.get(), onnx_buf_size)) {
        cerr << "IParser::parse() failed\n";
        int num_errors = parser_ptr->getNbErrors();
        cerr << "IParser::getNbErrors() = " << num_errors << '\n';
        for (int i = 0; i < num_errors; ++i) {
            auto err_ptr = parser_ptr->getError(i);
            if (err_ptr == nullptr) {
                cerr << "IParser::getError(" << i << ") = nullptr\n";
                continue;
            }
            cerr << "IParser::getError(" << i << "):\n";
            auto code = err_ptr->code();
            switch (code) {
            case ErrorCode::kSUCCESS:
                cerr << "code = kSUCCESS\n";
                break;
            case ErrorCode::kINTERNAL_ERROR:
                cerr << "code = kINTERNAL_ERROR\n";
                break;
            case ErrorCode::kMEM_ALLOC_FAILED:
                cerr << "code = kMEM_ALLOC_FAILED\n";
                break;
            case ErrorCode::kMODEL_DESERIALIZE_FAILED:
                cerr << "code = kMODEL_DESERIALIZE_FAILED\n";
                break;
            case ErrorCode::kINVALID_VALUE:
                cerr << "code = kINVALID_VALUE\n";
                break;
            case ErrorCode::kINVALID_GRAPH:
                cerr << "code = kINVALID_GRAPH\n";
                break;
            case ErrorCode::kINVALID_NODE:
                cerr << "code = kINVALID_NODE\n";
                break;
            case ErrorCode::kUNSUPPORTED_GRAPH:
                cerr << "code = kUNSUPPORTED_GRAPH\n";
                break;
            case ErrorCode::kUNSUPPORTED_NODE:
                cerr << "code = kUNSUPPORTED_NODE\n";
                break;
            default:
                cerr << "code = [[ UNKNOWN ]]\n";
                break;
            }
            auto msg = err_ptr->desc();
            cerr << "message = " << msg << '\n';
            cerr << "onnx node index = " << err_ptr->node() << '\n';
        }
        return 1;
    }
    builder_ptr->setMaxBatchSize(1);
    IBuilderConfig * builder_config_ptr = builder_ptr->createBuilderConfig();
    builder_config_ptr->setMaxWorkspaceSize(16 * 1024 * 1024);
    ICudaEngine * ptr = builder_ptr->buildEngineWithConfig(
        *netdef_ptr,
        *builder_config_ptr
    );
    if (ptr == nullptr) {
        cerr << "IBuilder::buildCudaEngine() failed\n";
        return 1;
    }
    cerr << "Looks OK!\n";
    return 0;
}
