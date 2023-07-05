import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt_structured import *
from modelutils import *

import sys
import os
import gc

sys.path.append('/nfs/homedirs/rachwan/Torch-Pruning/')

import torch_pruning as tp

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Perform garbage collection
        print("CUDA memory and garbage collection complete.")
    else:
        print("CUDA is not available on this system.")


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


# @torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print("Dense model inference time")
    batch = next(iter(dataloader))[0].cuda()
    model.cuda()
    model.eval()
    for _ in range(10):
        model(batch)
    num_iterations = 100  # Change this based on your needs
    timings = []

    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            output = model(batch)
        end_time = time.time()
        timings.append(end_time - start_time)

    # Convert to milliseconds
    timings_ms = np.array(timings) * 1000
    mean_time = np.mean(timings_ms)
    std_time = np.std(timings_ms)
    min_time = np.min(timings_ms)
    max_time = np.max(timings_ms)

    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min inference time: {min_time:.2f} ms")
    print(f"Max inference time: {max_time:.2f} ms")
    model.cpu()
    model.train()
    torchpruning = True
    print('Starting ...')
    import pdb
    # pdb.set_trace()
    with torch.no_grad():
        dtype = next(iter(model.parameters())).dtype

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.decoder.layers

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        layers[0] = layers[0].to(dev)
        model.to(dev)
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']

        print('Ready.')
        scores = {}
        for i in range(len(layers)):
            layer = layers[i].to(dev)

            subset = find_layers(layer)

            gpts = {}
            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            # model.cpu()
            # import copy
            # model_pruning = copy.deepcopy(model)
            # model.to(dev)
            for name in gpts:
                print(i, name)
                sparsity = args.sparsity
                scores[subset[name]] = gpts[name].get_scores(
                    sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
                )

                print('Pruning ...')

                # sparsity = args.sparsity
                # gpts[name].fasterprune(
                #     sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
                # )
                gpts[name].free()

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

            inps, outs = outs, inps
    # import pdb
    # pdb.set_trace()
    print('Torch_Pruning ...')
    ignored_layers = []
    iterative_steps = 1
    batch = next(iter(dataloader))
    example_inputs = batch[0].to(dev)
    imp = tp.importance.OBCImportance(scores, p=1, group_reduction='median')
    # imp = tp.importance.MagnitudeImportance(p=1)
    model.to(dev)

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 50272:
            ignored_layers.append(m)  # DO NOT prune the final classifier!
        elif isinstance(m, nn.Embedding):
            ignored_layers.append(m)
        elif 'project_out' in name:
            ignored_layers.append(m)
        elif 'k_proj' in name:
            ignored_layers.append(m)
    #
    # round_to = int(16 * (1 - args.sparsity))
    round_to = 12
    model.cpu()
    example_inputs.cpu()
    model.double()
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs.cpu(),
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=0.1,
        ignored_layers=ignored_layers,
        global_pruning=False,
        round_to=round_to
    )
    model.half()
    model.cuda()
    example_inputs.cuda()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(macs)
    for i in range(iterative_steps):
        # Get indices to prune without pruning
        indices = pruner.get_indices(args.sparsity, num_heads=round_to)
        # import pdb
        # pdb.set_trace()
        # pruner.step()
        # for m in model.modules():
        #     if hasattr(m, 'num_heads'):
        #         m.num_heads = round_to
        #         m.embed_dim = m.k_proj.out_features
        #         m.head_dim = m.embed_dim // m.num_heads
        #         m.scaling = m.head_dim ** -0.5
        # m.channels = m.conv.in_channels
        # model.config.hidden_size = model.model.decoder.embed_tokens.embedding_dim
        # model.config.word_embed_proj_dim = model.model.decoder.embed_tokens.embedding_dim
        print(model)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(macs)

    clear_cuda_memory()

    with torch.no_grad():
        dtype = next(iter(model.parameters())).dtype

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.decoder.layers

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        layers[0] = layers[0].to(dev)
        model.to(dev)
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']

        print('Ready.')
        scores = {}
        for i in range(len(layers)):
            layer = layers[i].to(dev)

            subset = find_layers(layer)

            gpts = {}
            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            # model.cpu()
            # import copy
            # model_pruning = copy.deepcopy(model)
            # model.to(dev)
            for name in gpts:

                print('Pruning ...')
                g1 = None
                g2 = None
                pruning_params = []
                for group in reversed(list(indices.keys())):
                    dep = group.has_layer(subset[name])
                    if dep is not None:
                        g1 = group
                        break  # Since you're looking for the last group, you can break once you find it

                # At this point, last_group will be the last
                # TODO: Maybe do one normal loop and one reversed loop
                if dep is None:
                    continue
                if 'out' in dep:
                    channel_to_prune = 'out'
                else:
                    channel_to_prune = 'in'

                pruning_params.append((g1, channel_to_prune))

                for group in indices.keys():
                    dep = group.has_layer(subset[name])
                    if dep is not None:
                        g2 = group
                        break  # Since you're looking for the last group, you can break once you find it

                # At this point, last_group will be the last
                # TODO: Maybe do one normal loop and one reversed loop
                if dep is None:
                    continue
                if 'out' in dep:
                    channel_to_prune = 'out'
                else:
                    channel_to_prune = 'in'

                pruning_params.append((g2, channel_to_prune))

                gpts[name].prune(
                    indices, pruning_params, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp,
                    blocksize=args.blocksize
                )
                gpts[name].free()

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

            inps, outs = outs, inps

    # # Final pruning using the pruner
    # for module in model.modules():
    #     # Check if any linear layer has 0s in the weight matrix
    #     if hasattr(module, 'weight') and torch.sum(module.weight) == 0:
    #         import pdb
    #         pdb.set_trace()
    model.eval()
    batch = next(iter(dataloader))[0].cuda()
    model.cuda()
    # model.double()
    # # def hook_fn(module, input, output):
    #     # print(module)
    #     # print(input)
    #     # print(module.weight.data)
    #     # print(module.bias if hasattr(module, 'bias') else print())
    #     print("Output of the layer: \n", output)
    #
    # hooks = [m.register_forward_hook(hook_fn) for name, m in model.named_modules() if
    #          'decoder.layers.1.fc1' in name or 'decoder.layers.0.fc2' in name or 'decoder.layers.0.final_layer_norm' in name]

    # for hook in hooks:
    #     hook.remove()

    # model.cpu()
    # for name, m in model.named_modules():
    #     if 'decoder.layers.10.fc2' in name:
    #         import copy
    #         bb = copy.deepcopy(m.bias)
    #         ww = copy.deepcopy(m.weight)
    #     if 'decoder.layers.10.final_layer_norm' in name:
    #         ln = copy.deepcopy(m.weight)
    #         lb = copy.deepcopy(m.bias)
    model.eval()
    pruner.prune(indices)
    # for name, m in model.named_modules():
    #     if 'decoder.layers.10.fc2' in name:
    #         # check if bias and weight are the same
    #         assert torch.equal(m.bias, bb)
    #         assert torch.equal(m.weight, ww)
    #     if 'decoder.layers.10.final_layer_norm' in name:
    #         assert torch.equal(m.weight, ln)
    #         assert torch.equal(m.bias, lb)
    print(model)

    for m in model.modules():
        if hasattr(m, 'num_heads'):
            m.num_heads = round_to
            m.embed_dim = m.k_proj.out_features
            m.head_dim = m.embed_dim // m.num_heads
            m.scaling = m.head_dim ** -0.5

            # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            #     return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            # m._shape = _shape

    # model.config.hidden_size = model.model.decoder.embed_tokens.embedding_dim
    # model.config.word_embed_proj_dim = model.model.decoder.embed_tokens.embedding_dim
    # model.cpu()
    # hooks = [m.register_forward_hook(hook_fn) for name, m in model.named_modules() if
    #          'decoder.layers.1.fc1' in name or 'decoder.layers.0.fc2' in name or 'decoder.layers.0.final_layer_norm' in name]
    for _ in range(10):
        model(batch)
    num_iterations = 100  # Change this based on your needs
    timings = []

    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            output = model(batch)
        end_time = time.time()
        timings.append(end_time - start_time)

    # Convert to milliseconds
    timings_ms = np.array(timings) * 1000
    mean_time = np.mean(timings_ms)
    std_time = np.std(timings_ms)
    min_time = np.min(timings_ms)
    max_time = np.max(timings_ms)

    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min inference time: {min_time:.2f} ms")
    print(f"Max inference time: {max_time:.2f} ms")
    # for hook in hooks:
    #     hook.remove()
    model.half()
    model.cuda()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(macs)
    # for module in model.modules():
    #     import pdb
    #     if isinstance(module, nn.Linear):
    #         pdb.set_trace()
    # model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        # import pdb
        # pdb.set_trace()
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)

        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)

        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        # import pdb
        # pdb.set_trace()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)

    model.train()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # pdb.set_trace()

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
