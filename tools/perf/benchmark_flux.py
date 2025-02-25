import argparse
import torch
from diffusers import FluxPipeline
from utils import print_bm_results, add_csv_result
import os

import torch_migraphx

torch._dynamo.reset()

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument('--num_steps',
                    type=int,
                    default=50,
                    help='Number of steps to run unet')

parser.add_argument('--prompts',
                    nargs='*',
                    default=["A cat holding a sign that says hello world"],
                    help='Prompts to use as input')

parser.add_argument('--prompt2',
                    nargs='*',
                    default=None,
                    help='Prompts to use as input')

parser.add_argument('--model_repo',
                    type=str,
                    default='black-forest-labs/FLUX.1-dev',
                    help='Huggingface repo path')

parser.add_argument('--bf16',
                    action='store_true',
                    help='Load bf16 version of the pipeline')

parser.add_argument("-d",
                    "--image-height",
                    type=int,
                    default=1024,
                    help="Output Image height, default 1024")

parser.add_argument("-w",
                    "--image-width",
                    type=int,
                    default=1024,
                    help="Output Image width, default 1024")

parser.add_argument("-i",
                    "--iterations",
                    type=int,
                    default=100,
                    help="Iterations for benchmarking")

parser.add_argument('--inductor', 
                    action='store_true', 
                    default=False,
                    help='Perform benchmark with torch inductor backend (default settings)')

parser.add_argument('--csv', 
                    type=str, 
                    default="",
                    help='Add perf results to a csv file')


def benchmark_module(pipe, args) -> float:
    pipe(prompt=args.prompts,
                 height=args.image_height,
                 width=args.image_width,
                 guidance_scale=3.5,
                 num_inference_steps=args.num_steps,
                 max_sequence_length=512)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.iterations):
        pipe(prompt=args.prompts,
            height=args.image_height,
            width=args.image_width,
            guidance_scale=3.5,
            num_inference_steps=args.num_steps,
            max_sequence_length=512)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / args.iterations


def benchmark_flux_model(args):

    model_names = [] 
    times = []

    torch_dtype = torch.float32
    dtype = "fp32"
    options = {}
    if args.bf16:
        dtype = "fp16"
        options["bf16"] = True
        torch_dtype = torch.bfloat16

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype)
    pipe = pipe.to("cuda")

    torch_res = benchmark_module(pipe, args)
    model_names.append("Torch Model")
    times.append(torch_res)

    if args.inductor:
        torch._dynamo.reset()

        pipe.text_encoder = torch.compile(pipe.text_encoder)
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
        pipe.transformer = torch.compile(pipe.transformer)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder)

        inductor_res = benchmark_module(pipe, args)

        model_names.append("Torch Inductor")
        times.append(inductor_res)

    del pipe

    if "migraphx" in torch._dynamo.list_backends():
        torch._dynamo.reset()

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)
        pipe = pipe.to("cuda")

        pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx', options=options)
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, backend='migraphx', options=options)
        pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', options=options)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx', options=options)

        mgx_dynamo_res = benchmark_module(pipe, args)
        
        model_names.append("MIGraphX Dynamo")
        times.append(mgx_dynamo_res)
        del pipe

    print_bm_results(model_names, times, 1)

    if args.csv:
        add_csv_result(args.csv, "FLUX.1-dev", model_names, times, 1, dtype)


if __name__ == '__main__':
    args = parser.parse_args()
    benchmark_flux_model(args)
