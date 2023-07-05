import torch
import torch.nn as nn
import typing

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency, sparsegpt_structured


class MetaPruner:
    """
        Meta Pruner for structural pruning.

        Args:
            model (nn.Module): A to-be-pruned model
            example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            importance (Callable): importance estimator.
            global_pruning (bool): enable global pruning.
            ch_sparsity (float): global channel sparisty.
            ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
            iterative_steps (int): number of steps for iterative pruning.
            iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            ignored_layers (List[nn.Module]): ignored modules.

            round_to (int): channel rounding (Channels to keep) .
            customized_pruners (dict): a dict containing module-pruner pairs.
            unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.
            root_module_types (list): types of prunable modules.
            output_transform (Callable): A function to transform network outputs.
        """

    def __init__(
            self,
            # Basic
            model: nn.Module,
            example_inputs: torch.Tensor,
            importance: typing.Callable,
            global_pruning: bool = False,
            semi_global_pruning: bool = False,
            ch_sparsity: float = 0.5,  # channel/dim sparsity
            ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
            ch_group_sparsity_dict: typing.Dict[nn.Module, float] = None,
            max_ch_sparsity: float = 1.0,
            iterative_steps: int = 1,  # for iterative pruning
            iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
            ignored_layers: typing.List[nn.Module] = None,
            module_list=None,

            # Advanced
            round_to: int = None,  # round channels to 8x, 16x, ...
            channel_groups: typing.Dict[nn.Module, int] = dict(),  # for grouped channels.
            customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
            # pruners for customized layers
            unwrapped_parameters: typing.List[nn.Parameter] = None,  # unwrapped nn.Parameters like ViT.pos_emb
            root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],
            # root module for each group
            output_transform: typing.Callable = None,

            # Gradient-based
            crit=None,
            Loss=None,
            dataloader=None,
            backward_needed=True,
            pruning_batch_size=64,
            iterations=1,

            # Yolo
            yolo=False,
            scaler=None

    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.ch_group_sparsity_dict = ch_group_sparsity_dict if ch_group_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning
        self.semi_global_pruning = semi_global_pruning

        self.channel_groups = channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        self.crit = crit
        self.Loss = Loss
        self.pruning_batch_size = pruning_batch_size
        self.backward_needed = backward_needed
        self.yolo = yolo
        self.scaler = scaler
        self.dataloader = dataloader
        self.module_list = module_list
        self.iterations = iterations

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
        )

        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The customized channel sparsity for different layers
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # detect group convs
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups

        # detect group norms
        self.round_to = 0 if self.round_to==None else round_to
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.round_to = m.num_groups

        #         self.channel_groups[m] = m.num_groups

        # detect transformer heads
        #TODO this isnt working because when we have both group norm and attention then there is 2 different round-to
        import diffusers
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        for m in self.model.modules():
            if hasattr(m, 'attn_num_head_channels'):
                self.round_to = lcm(m.attn_num_head_channels, self.round_to)

        if self.global_pruning:
            initial_total_channels = 0
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
                ch_groups = self.get_channel_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )
                initial_total_channels += (self.DG.get_out_channels(
                    group[0][0].target.module) // ch_groups)
            self.initial_total_channels = initial_total_channels

    def pre_prune(self, pruning_batch_size=64, iterations=1, yolo=False):
        pass
    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
            self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass

    def step(self, interactive=False):
        self.current_step += 1
        if self.global_pruning:
            if self.backward_needed == True:
                self.pre_prune(pruning_batch_size=self.pruning_batch_size, iterations=self.iterations, yolo=self.yolo)
            if self.semi_global_pruning:
                self.semi_prune_global()
            elif interactive:
                return self.prune_global()
            else:
                for group in self.prune_global():
                    group.prune()
        else:
            if interactive:
                return self.prune_local()
            else:
                for group in self.prune_local():
                    group.prune()

    def get_indices(self, sparsity, num_heads):
        indices = {}
        for idx, group in self.indices(sparsity, num_heads):
            indices[group] = idx
        return indices

    def estimate_importance(self, group, ch_groups=1):
        return self.importance(group, ch_groups=ch_groups)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if function.is_out_channel_pruner(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)

                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif function.is_in_channel_pruner(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            # and function.is_out_channel_pruner(dep.handler):
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping

    def prune(self, indices):
        num_pruned = 0
        for group in self.prune_index(indices):
            # print(group)
            # if num_pruned < 10:
            group.prune()
                # num_pruned += 1
            # else:
            #     break

    def prune_index(self, indices):

        for group in indices.keys():
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                pruning_idxs = indices[group]
                # import pdb
                # pdb.set_trace()
                g = self.DG.get_pruning_group(module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(g):
                    yield g


    def indices(self, sparsity, num_heads):
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                if self.backward_needed == True:
                    # Suboptimal since it recalculates scores for every layer
                    self.pre_prune(pruning_batch_size=self.pruning_batch_size, iterations=self.iterations,
                                   yolo=self.yolo)
                # TODO: Here the importance should be extracted from the sparsegpt one
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if len(imp) == 0:
                    yield torch.tensor([]), group
                    print(group)
                    continue
                current_channels = self.DG.get_out_channels(module)
                # TODO: Fix this, not harcoded
                target_sparsity = sparsity
                n_pruned = int(self.layer_init_out_ch[module] * target_sparsity)

                if self.round_to:
                    # Calculate the remainder after dividing (current_channels - n_pruned) by self.round_to
                    remainder = (current_channels - n_pruned) % self.round_to

                    # If there is a remainder, add it to n_pruned to make (current_channels - n_pruned) a multiple of self.round_to
                    if remainder:
                        n_pruned += remainder

                if ch_groups > 1:
                    imp = imp[:len(imp) // ch_groups]

                def sort_indices_by_group_products(tensor, group_size):
                    def group_elements(tensor, group_size):
                        if group_size <= 0:
                            raise ValueError("Group size must be a positive integer.")

                        grouped_elements = []
                        for i in range(0, len(tensor), group_size):
                            group = tensor[i:i + group_size]
                            grouped_elements.append(group)

                        return grouped_elements

                    def product_of_groups(grouped_list):
                        products = []
                        for group in grouped_list:
                            product = 1
                            for element in group:
                                product *= element
                            products.append(product)

                        return products

                    grouped_list = group_elements(tensor, group_size)
                    products_list = product_of_groups(grouped_list)
                    sorted_indices = sorted(range(len(products_list)), key=lambda i: products_list[i])

                    original_indices = []
                    for index in sorted_indices:
                        group_original_indices = list(range(index * group_size, (index * group_size) + group_size))
                        # Clip the indices to the length of the original tensor
                        group_original_indices = [i for i in group_original_indices if i < len(tensor)]
                        original_indices.extend(group_original_indices)

                    return original_indices
                import pdb
                for name, m in self.model.named_modules():
                    if m == module:
                        break
                if any(substring in name for substring in ('k_proj', 'v_proj', 'q_proj', 'out_proj')):
                    imp_argsort = torch.tensor(sort_indices_by_group_products(imp, num_heads))
                else:
                    imp_argsort = torch.argsort(imp)
                # TODO: Here we should call the sparsegpt algorithm with the given mask to fix the other weights before structured pruning
                pruning_idxs = imp_argsort[:(n_pruned // ch_groups)]
                if ch_groups > 1:
                    group_size = current_channels // ch_groups
                    pruning_idxs = torch.cat(
                        [pruning_idxs + group_size * i for i in range(ch_groups)], 0)

                yield pruning_idxs.to(torch.long), group

    def prune_local_index(self, index, nb_parameters=10):
        if self.current_step >= self.iterative_steps:
            return

        layers = [str(i) for i in range(25)]
        layers.reverse()

        for idx, group in enumerate(self.get_all_groups()):
            # check pruning rate
            if idx == index:
                deps = "".join([str(dep) for dep, _ in group])
                import re

                id = re.findall(r"model\.(\d+)", str(deps))
                id = list(set(id))

                if self._check_sparsity(group):
                    module = group[0][0].target.module
                    pruning_fn = group[0][0].handler

                    ch_groups = self.get_channel_groups(group)
                    if self.backward_needed == True:
                        # Suboptimal since it recalculates scores for every layer
                        self.pre_prune(pruning_batch_size=self.pruning_batch_size, iterations=self.iterations,
                                       yolo=self.yolo)
                    imp = self.estimate_importance(group, ch_groups=ch_groups)
                    current_channels = self.DG.get_out_channels(module)
                    target_sparsity = self.get_target_sparsity(module)
                    n_pruned = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (1 - target_sparsity)
                    )
                    n_pruned = nb_parameters

                    if self.round_to:
                        n_pruned = n_pruned - (n_pruned % self.round_to)

                    if n_pruned <= 0:
                        continue
                    if ch_groups > 1:
                        imp = imp[:len(imp) // ch_groups]
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs = imp_argsort[:(n_pruned // ch_groups)]
                    if ch_groups > 1:
                        group_size = current_channels // ch_groups
                        pruning_idxs = torch.cat([pruning_idxs + group_size * i for i in range(ch_groups)], 0)
                    group = self.DG.get_pruning_group(
                        module, pruning_fn, pruning_idxs.tolist())
                    if self.DG.check_pruning_group(group):
                        group.exec()
                    break
        return id

    def prune_local(self):
        # import pdb
        # pdb.set_trace()
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                if self.backward_needed == True:
                    # Suboptimal since it recalculates scores for every layer
                    self.pre_prune(pruning_batch_size=self.pruning_batch_size, iterations=self.iterations,
                                   yolo=self.yolo)
                #TODO: Here the importance should be extracted from the sparsegpt one
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                if self.round_to:
                    #TODO: This might be logically incorrect, see indices method
                    n_pruned = n_pruned - (n_pruned % self.round_to)

                if n_pruned <= 0:
                    continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                imp_argsort = torch.argsort(imp)
                #TODO: Here we should call the sparsegpt algorithm with the given mask to fix the other weights before structured pruning
                pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                if ch_groups > 1:
                    group_size = current_channels//ch_groups
                    pruning_idxs = torch.cat(
                        [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self):
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                global_importance.append((group, ch_groups, imp))

        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)

        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                yield group

    def semi_prune_global(self, num_splits=5, local=False):
        # print([m for m in self.DG.module2node.keys() if isinstance(m, tuple(self.root_module_types))])
        # print(len([m for m in self.DG.module2node.keys() if isinstance(m, tuple(self.root_module_types))]))
        # breakpoint()
        if self.current_step >= self.iterative_steps:
            return

        global_importance = []
        for group in self.get_all_groups():
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if ch_groups > 1:
                    imp = imp[:len(imp) // ch_groups]
                global_importance.append((group, ch_groups, imp))

        def split_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        def split_list_specific(lst, chunk_list):
            j = 0
            for i in range(len(chunk_list)):
                yield lst[j:i + 1]
                j += i + 1

        def split_list_new(lst):
            backbone = []
            neck = []
            head = []
            indices = []
            backbone_indices = []
            neck_indices = []
            head_indices = []

            for idx, (group, ch_groups, imp) in enumerate(global_importance):
                deps = "".join([str(dep) for dep, _ in group])
                if 'model.24' in deps:
                    head.append((group, ch_groups, imp))
                    head_indices.append(idx)
                elif 'model.10' in deps or 'model.11' in deps or 'model.12' in deps or 'model.13' in deps or 'model.14' in deps or 'model.15' in deps or 'model.16' in deps or 'model.17' in deps or 'model.18' in deps or 'model.19' in deps or 'model.20' in deps or 'model.21' in deps or 'model.22' in deps or 'model.23' in deps:
                    neck.append((group, ch_groups, imp))
                    neck_indices.append(idx)
                else:
                    backbone.append((group, ch_groups, imp))
                    backbone_indices.append(idx)

                # for dep, _ in group:
                #     if '24' in str(dep):
                #         indices.append(idx)
                #         head.append((group, ch_groups, imp))
                #         break
            return backbone, neck, head, backbone_indices, neck_indices, head_indices

        imp_list = []
        if not local:
            len_splits = int(len(global_importance) / num_splits)
        else:
            len_splits = 1
        print(len_splits)
        # splits = list(split_list_specific(global_importance, [46,3]))
        splits = []
        backbone, neck, head, backbone_indices, neck_indices, head_indices = split_list_new(global_importance)
        splits.append(backbone)
        splits.append(neck)
        splits.append(head)
        print(len(splits))
        print("lol")
        print(len(splits[0]))
        print(len(splits[1]))
        print(len(splits[2]))
        # breakpoint()
        for split in splits:
            imp_list.append(torch.cat([local_imp[-1] for local_imp in split], dim=0))

        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned_list = []
        for im in imp_list:
            n_pruned = len(im) - int(
                len(im) *
                (1 - target_sparsity)
            )
            if n_pruned <= 0:
                return
            n_pruned_list.append(n_pruned)

        topk_imp_list = []
        for i, im in enumerate(imp_list):
            topk_imp, _ = torch.topk(im, k=n_pruned_list[i], largest=False)
            topk_imp_list.append(topk_imp)

        # semi-global pruning through thresholding
        thres_list = []
        for topk_imp in topk_imp_list:
            thres_list.append(topk_imp[-1])
        idx = 0
        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            # thres_idx = int(idx / (len(global_importance) / len(splits)))
            if idx in backbone_indices:
                thres_idx = 0
            elif idx in neck_indices:
                thres_idx = 1
            else:
                thres_idx = 2

            print(thres_idx)
            pruning_indices = (imp <= thres_list[thres_idx]).nonzero().view(-1)

            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module) // ch_groups
                pruning_indices = torch.cat([pruning_indices + group_size * i for i in range(ch_groups)], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                group.exec()
            idx += 1
