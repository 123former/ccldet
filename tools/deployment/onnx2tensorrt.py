import os
import tensorrt as trt
import pdb
TRT_LOGGER = trt.Logger()
model_path = '/home/f523/guazai/disk3/shangxiping/mmrotate/model_sxp_576_576.onnx'
engine_file_path = "/home/f523/guazai/disk3/shangxiping/mmrotate/model_sxp_576_576.trt"
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # batchsize=1

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) \
        as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)

    # builder.max_workspace_size = 1 << 28
    builder.max_batch_size = 1
    if not os.path.exists(model_path):
        print('ONNX file {} not found.'.format(model_path))
        exit(0)
    print('Loading ONNX file from path {}...'.format(model_path))
    with open(model_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    network.get_input(0).shape = [1, 3, 576, 576]
    print('Completed parsing of ONNX file')

    # last_layer = network.get_layer(network.num_layers - 1)
    # network.mark_output(last_layer.get_output(0))
    # pdb.set_trace()

    engine = builder.build_engine(network, config)

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())