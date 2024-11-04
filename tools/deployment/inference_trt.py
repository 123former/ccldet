import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import os
import tensorrt as trt
import pdb
from tqdm import tqdm
from argparse import ArgumentParser

# model_path = 'FashionMNIST.onnx'
# engine_file_path = "/home/f523/guazai/disk3/shangxiping/mmrotate/model_sxp_576_576_112.trt"


def resize_img_keep_ratio(img, target_size):
    old_size = img.shape[0:2]
    # ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new, ratio


def preprocess(img_data):
    img_data, ratio = resize_img_keep_ratio(img_data, (576, 576))
    # img_data = cv2.flip(img_data, 0)
    img_data = np.transpose(img_data, (2, 0, 1))
    # img_data = cv2.resize(img_data, (576, 576), interpolation=cv2.INTER_AREA)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    return norm_img_data, ratio


def draw(img, xscale, yscale, pred, out_path, thre=0.5):
    img_ = img.copy()
    bboxes = pred[0]
    labels = pred[1]

    for i in range(bboxes.shape[0]):
        detect = bboxes[i, :4]
        conf = bboxes[i, -1]
        cls = labels[i]
        if cls == 4:
            continue
        if cls == 3:
            continue
        if conf < thre:
            continue

        detect = [int((detect[0] - 117) * xscale), int((detect[1]) * yscale),
                  int((detect[2] - 117) * xscale), int((detect[3]) * yscale)]
        # pdb.set_trace()
        # img_ = cv2.rectangle(img, (detect[0], 3500-detect[3]), (detect[2], 3500-detect[1]), (0, 255, 255), 5)
        # cv2.putText(img, '%s %.3f' % (str(cls), conf), (detect[0], 3500-detect[3] + 10),
        #             color=(0, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=4)
        img_ = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0, 255, 255), 5)
        cv2.putText(img, '%s %.3f' % (str(cls), conf), (detect[0], detect[1] + 100),
                    color=(0, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=4)
        cv2.imwrite(out_path, img_)
    return img_


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def test_trt(img_path, out_path, context, bindings, inputs, outputs, stream, merge=0):
    img = cv2.imread(img_path)
    input_data, ratio = preprocess(img)
    image = input_data.reshape([1, 3, 576, 576])

    inputs[0].host = image
    # 开始推理
    result = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    labels_merge, bboxes_merge = result[0], result[1].reshape([100, 5])
    img_ = draw(img, 1 / ratio, 1 / ratio, [bboxes_merge, labels_merge], out_path)
    # img_ = draw(img, 1 / ratio, 1 / ratio, [bboxes_merge, labels_merge], out_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


# create_server()
# create_client()
def main(args):
    engine_file_path = args.checkpoint
    img_list = os.listdir(args.img_dir)
    if not os.path.exists(args.out_file):
        os.makedirs(args.out_file)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)

    for img in tqdm(img_list):
        if '.jpg' not in img:
            continue
        img_path = os.path.join(args.img_dir, img)
        out_path = os.path.join(args.out_file, img)

        test_trt(img_path, out_path, context, bindings, inputs, outputs, stream, merge=0)

if __name__ == '__main__':
    args = parse_args()
    main(args)