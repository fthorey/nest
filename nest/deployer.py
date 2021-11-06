import os
import statistics
import time
from collections import Counter
import torch
from box import Box
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from nest import deployer
from yolov5 import export

IMG_SIZE = (640, 480)
torch_type_to_triton_type = {
    torch.bool: "TYPE_BOOL",
    torch.int8: "TYPE_INT8",
    torch.int16: "TYPE_INT16",
    torch.int32: "TYPE_INT32",
    torch.int64: "TYPE_INT64",
    torch.uint8: "TYPE_UINT8",
    torch.float16: "TYPE_FP16",
    torch.float32: "TYPE_FP32",
    torch.float64: "TYPE_FP64",
}

CONFIG_TEMPLATE = r"""
name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}
input [
    {spec_inputs}
]
output [
    {spec_outputs}
]
{dynamic_batching}
{model_optimizations}
instance_group [
    {{
        count: {engine_count}
        kind: KIND_GPU
        gpus: [ {gpu_list} ]
    }}
]"""

INPUT_TEMPLATE = r"""
{{
    name: "input__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

OUTPUT_TEMPLATE = r""" 
{{
    name: "output__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

MODEL_OPTIMIZATION_TEMPLATE = r"""
optimization {{
  cuda {{
    graphs: {capture_cuda_graph}
  }}
}}"""


def remove_empty_lines(text):
    """removes empty lines from text, returns the result"""
    ret = "".join([s for s in text.strip().splitlines(True) if s.strip()])
    return ret


def get_cfg(model_name, method="trace", max_batch_size=2, triton_no_cuda=True):
    cfg = Box()
    cfg.method = method
    cfg.triton_no_cuda = triton_no_cuda
    cfg.ts_trace = True if method == "trace" else False
    cfg.ts_script = True
    cfg["triton-model-name"] = model_name
    cfg["triton-model-version"] = 0
    cfg["triton-server-url"] = "localhost:8001"
    cfg["triton-max-batch-size"] = max_batch_size
    cfg["triton-dyn-batching-delay"] = 0
    cfg["triton-engine-count"] = 1
    cfg["capture-cuda-graph"] = 0
    cfg["model_arguments"] = None
    cfg["save-dir"] = f"/workdir/triton_models"
    return cfg


class DeployerLibrary:
    def __init__(self, args):
        self.args = args
        self.platform = None

    def set_platform(self, platform):
        """sets the platform
        :: platform :: "pytorch_libtorch" or "onnxruntime_onnx" or "tensorrt_plan"
        """
        self.platform = platform

    def prepare_inputs(self, dataloader, device):
        """load sample inputs to device"""
        inputs = []
        for batch in dataloader:
            batch = batch[0:1]
            if type(batch) is torch.Tensor:
                batch_d = batch.to(device)
                batch_d = (batch_d,)
                inputs.append(batch_d)
            else:
                batch_d = []
                for x in batch:
                    assert type(x) is torch.Tensor, "input is not a tensor"
                    batch_d.append(x.to(device))
                batch_d = tuple(batch_d)
                inputs.append(batch_d)
        return inputs

    def get_list_of_shapes(self, l, fun):
        """returns the list of min/max shapes, depending on fun
        :: l :: list of tuples of tensors
        :: fun :: min or max
        """
        tensor_tuple = l[0]
        shapes = [list(x.shape) for x in tensor_tuple]
        for tensor_tuple in l:
            assert len(tensor_tuple) == len(
                shapes
            ), "tensors with varying shape lengths are not supported"
            for i, x in enumerate(tensor_tuple):
                for j in range(len(x.shape)):
                    shapes[i][j] = fun(shapes[i][j], x.shape[j])
        return shapes  # a list of shapes

    def get_tuple_of_min_shapes(self, l):
        """returns the tuple of min shapes
        :: l :: list of tuples of tensors"""
        shapes = self.get_list_of_shapes(l, min)
        min_batch = 1
        shapes = [[min_batch, *shape[1:]] for shape in shapes]
        shapes = tuple(shapes)
        return shapes  # tuple of min shapes

    def get_tuple_of_max_shapes(self, l):
        """returns the tuple of max shapes
        :: l :: list of tuples of tensors"""
        shapes = self.get_list_of_shapes(l, max)
        max_batch = max(2, shapes[0][0])
        shapes = [[max_batch, *shape[1:]] for shape in shapes]
        shapes = tuple(shapes)
        return shapes  # tuple of max shapes

    def get_tuple_of_opt_shapes(self, l):
        """returns the tuple of opt shapes
        :: l :: list of tuples of tensors"""
        counter = Counter()
        for tensor_tuple in l:
            shapes = [tuple(x.shape) for x in tensor_tuple]
            shapes = tuple(shapes)
            counter[shapes] += 1
        shapes = counter.most_common(1)[0][0]
        return shapes  # tuple of most common occuring shapes

    def get_tuple_of_dynamic_shapes(self, l):
        """returns a tuple of dynamic shapes: variable tensor dimensions
        (for ex. batch size) occur as -1 in the tuple
        :: l :: list of tuples of tensors"""
        tensor_tuple = l[0]
        shapes = [list(x.shape) for x in tensor_tuple]
        for tensor_tuple in l:
            err_msg = "tensors with varying shape lengths are not supported"
            assert len(tensor_tuple) == len(shapes), err_msg
            for i, x in enumerate(tensor_tuple):
                for j in range(len(x.shape)):
                    if shapes[i][j] != x.shape[j] or j == 0:
                        shapes[i][j] = -1
        shapes = tuple(shapes)
        return shapes  # tuple of dynamic shapes

    @torch.no_grad()
    def run_models(self, models, inputs):
        """run the models on inputs, return the outputs and execution times"""
        ret = []
        for model in models:
            # torch.cuda.synchronize()
            time_start = time.time()
            outputs = []
            for x in inputs:
                with torch.no_grad():
                    output = model(*x)
                if type(output) is torch.Tensor:
                    output = [output]
                outputs.append(output)
            # torch.cuda.synchronize()
            time_end = time.time()
            t = time_end - time_start
            ret.append(outputs)
            ret.append(t)
        return ret

    def compute_errors(self, outputs_A, outputs_B):
        """returns the list of L_inf errors computed over every single output tensor"""
        Linf_errors = []
        for output_A, output_B in zip(outputs_A, outputs_B):
            for x, y in zip(output_A, output_B):
                for idx in range(len(x)):
                    error = (x[idx] - y[idx]).norm(float("inf")).item()
                    Linf_errors.append(error)
        return Linf_errors

    def print_errors(self, Linf_errors):
        """print various statistcs of Linf errors"""
        print()
        print("conversion correctness test results")
        print("-----------------------------------")
        print("maximal absolute error over dataset (L_inf): ", max(Linf_errors))
        print()
        print("average L_inf error over output tensors: ", statistics.mean(Linf_errors))
        print(
            "variance of L_inf error over output tensors: ",
            statistics.variance(Linf_errors),
        )
        print(
            "stddev of L_inf error over output tensors: ", statistics.stdev(Linf_errors)
        )
        print()

    def write_config(
        self, config_filename, input_shapes, input_types, output_shapes, output_types
    ):
        """writes Triton config file
        :: config_filename :: the file to write the config file into
        :: input_shapes :: tuple of dynamic shapes of the input tensors
        :: input_types :: tuple of torch types of the input tensors
        :: output_shapes :: tuple of dynamic shapes of the output tensors
        :: output_types :: tuple of torch types of the output tensors
        """
        assert self.platform is not None, "error - platform is not set"

        config_template = CONFIG_TEMPLATE
        input_template = INPUT_TEMPLATE
        optimization_template = MODEL_OPTIMIZATION_TEMPLATE

        spec_inputs = r""""""
        for i, (shape, typ) in enumerate(zip(input_shapes, input_types)):
            d = {
                "num": str(i),
                "type": torch_type_to_triton_type[typ],
                "dims": str([1])
                if len(shape) == 1
                else str(list(shape)[1:]),  # first dimension is the batch size
            }
            d["reshape"] = "reshape: { shape: [ ] }" if len(shape) == 1 else ""
            spec_inputs += input_template.format_map(d)
        spec_inputs = spec_inputs[:-1]

        output_template = OUTPUT_TEMPLATE
        spec_outputs = r""""""
        for i, (shape, typ) in enumerate(zip(output_shapes, output_types)):
            d = {
                "num": str(i),
                "type": torch_type_to_triton_type[typ],
                "dims": str([1])
                if len(shape) == 1
                else str(list(shape)[1:]),  # first dimension is the batch size
            }
            d["reshape"] = "reshape: { shape: [ ] }" if len(shape) == 1 else ""
            spec_outputs += output_template.format_map(d)
        spec_outputs = spec_outputs[:-1]

        batching_str = ""
        max_batch_size = self.args.triton_max_batch_size

        if self.args.triton_dyn_batching_delay > 0:
            # Use only full and half full batches
            pref_batch_size = [int(max_batch_size / 2.0), max_batch_size]

            batching_str = r"""
dynamic_batching {{
    preferred_batch_size: [{0}]
    max_queue_delay_microseconds: {1}
}}""".format(
                ", ".join([str(x) for x in pref_batch_size]),
                int(self.args.triton_dyn_batching_delay * 1000.0),
            )

        d = {"capture_cuda_graph": str(self.args.capture_cuda_graph)}
        optimization_str = optimization_template.format_map(d)

        config_values = {
            "model_name": self.args.triton_model_name,
            "platform": self.platform,
            "max_batch_size": max_batch_size,
            "spec_inputs": spec_inputs,
            "spec_outputs": spec_outputs,
            "dynamic_batching": batching_str,
            "model_optimizations": optimization_str,
            "gpu_list": ", ".join([str(x) for x in range(torch.cuda.device_count())]),
            "engine_count": self.args.triton_engine_count,
        }

        # write config
        with open(config_filename, "w") as file:
            final_config_str = config_template.format_map(config_values)
            final_config_str = remove_empty_lines(final_config_str)
            file.write(final_config_str)


class Deployer(object):
    def __init__(self, args):
        self.args = args
        self.lib = DeployerLibrary(args)

    def deploy(self, dataloader, model):
        """deploy the model and test for correctness with dataloader"""
        if self.args.ts_script or self.args.ts_trace:
            self.lib.set_platform("pytorch_libtorch")
            print(
                "deploying model "
                + self.args.triton_model_name
                + " in format "
                + self.lib.platform
            )
            self.to_triton_torchscript(dataloader, model)
        elif self.args.onnx:
            self.lib.set_platform("onnxruntime_onnx")
            print(
                "deploying model "
                + self.args.triton_model_name
                + " in format "
                + self.lib.platform
            )
            self.to_triton_onnx(dataloader, model)
        else:
            assert False, "error"
        print("done")

    def to_triton_onnx(self, dataloader, model):
        """export the model to onnx and test correctness on dataloader"""
        import onnx
        import onnxruntime

        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate outputs
        outputs = []
        for input in inputs:
            with torch.no_grad():
                output = model(*input)
            if type(output) is torch.Tensor:
                output = [output]
            outputs.append(output)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # get input names
        rng = range(len(input_types))
        input_names = ["input__" + str(num) for num in rng]

        # get output names
        rng = range(len(output_types))
        output_names = ["output__" + str(num) for num in rng]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.onnx")

        # get indices of dynamic input and output shapes
        dynamic_axes = {}
        for input_name, input_shape in zip(input_names, input_shapes):
            dynamic_axes[input_name] = [i for i, x in enumerate(input_shape) if x == -1]
        for output_name, output_shape in zip(output_names, output_shapes):
            dynamic_axes[output_name] = [
                i for i, x in enumerate(output_shape) if x == -1
            ]

        # export the model
        assert not model.training, "internal error - model should be in eval() mode! "
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs[0],
                final_model_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
            )

        # syntactic error check
        converted_model = onnx.load(final_model_path)
        # check that the IR is well formed
        onnx.checker.check_model(converted_model)

        # load the model
        session = onnxruntime.InferenceSession(final_model_path, None)

        class ONNX_model:
            def __init__(self, session, input_names, device):
                self.session = session
                self.input_names = input_names

            def to_numpy(self, tensor):
                return (
                    tensor.detach().cpu().numpy()
                    if tensor.requires_grad
                    else tensor.cpu().numpy()
                )

            def __call__(self, *inputs):
                inp = [
                    (input_name, inputs[i])
                    for i, input_name in enumerate(self.input_names)
                ]
                inp = {input_name: self.to_numpy(x) for input_name, x in inp}
                outputs = self.session.run(None, inp)
                outputs = [torch.from_numpy(output) for output in outputs]
                outputs = [output.to(device) for output in outputs]
                if len(outputs) == 1:
                    outputs = outputs[0]
                return outputs

        # switch to eval mode
        model_onnx = ONNX_model(session, input_names, device)

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        models = (model, model_onnx)
        outputs, time_model, outputs_onnx, time_model_onnx = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Linf_errors = self.lib.compute_errors(outputs, outputs_onnx)
        self.lib.print_errors(Linf_errors)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of onnx model: ", time_model_onnx, "seconds")
        print()

        # write Triton config
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )

    def to_triton_torchscript(self, dataloader, model):
        """export the model to torchscript and test correctness on dataloader"""
        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.pt")

        # convert the model
        with torch.no_grad():
            model_ts = torch.jit.trace(model, inputs[0], strict=False)

        # save the model
        torch.jit.save(model_ts, final_model_path)

        # load the model
        model_ts = torch.jit.load(final_model_path)
        model_ts.eval()  # WAR for bug : by default, model_ts gets loaded in training mode

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        assert (
            not model_ts.training
        ), "internal error - converted model should be in eval() mode! "
        models = (model, model_ts)
        outputs, time_model, outputs_ts, time_model_ts = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Linf_errors = self.lib.compute_errors(outputs, outputs_ts)
        self.lib.print_errors(Linf_errors)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of ts model: ", time_model_ts, "seconds")
        print()

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # now we build the config for Triton
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )
        print("-" * 50)
        print(f"Successfully dumped in {config_filename}")


class YoloDataset(Dataset):
    def __init__(self, size=(640, 640)):
        self.size = size

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.zeros(3, *self.size).float()


def get_model(device="cpu", batch_size=1, imgsz=(480, 640)):
    torch.hub.load("ultralytics/yolov5", "yolov5s")
    weights = "/root/.cache/torch/hub/ultralytics_yolov5_master/yolov5s.pt"
    device = export.select_device(device)
    model = export.attempt_load(weights, map_location=device, inplace=True, fuse=True)
    im = torch.zeros(batch_size, 3, *imgsz).to(
        device
    )  # image size(1,3,320,192) BCHW iDetection
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, export.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = export.SiLU()
        elif isinstance(m, export.Detect):
            m.inplace = True
            m.onnx_dynamic = False
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(im)  # dry runs
    return model


class Wrapper(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = get_model(imgsz=size)

    def forward(self, x):
        return self.model(x)[0]


def to_triton(method="trace", max_batch_size=1, size=(480, 640)):
    m = Wrapper(size)
    ds = YoloDataset(size)
    dl = DataLoader(
        ds,
        batch_size=max_batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    deploy_cfg = deployer.get_cfg(
        "yolovs", method=method, max_batch_size=max_batch_size
    )
    dep = deployer.Deployer(deploy_cfg)
    dep.deploy(dl, m)
