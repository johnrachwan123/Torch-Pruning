from .metapruner import MetaPruner
import torch

class GradientPruner(MetaPruner):

    def pre_prune(self, pruning_batch_size=64, iterations=1, yolo=False):
        self.model.cuda()
        for i in range(iterations):
            img, target = next(iter(self.dataloader))
            if self.crit == "SNIP":
                outputs = self.model(img[:pruning_batch_size].cuda())
                loss = self.Loss(outputs, target[:pruning_batch_size].cuda())
                loss.backward()
            elif self.crit == 'SYNFLOW':
                input_dim = list(img[0, :].shape)
                out = self.model(torch.ones([1] + input_dim).cuda())
                out.sum().backward()
            elif self.crit == 'CROP':
                weights = []
                for layer in self.model.modules():
                    if hasattr(layer, 'weight') and layer.weight.requires_grad == True:
                        weights.append(layer.weight)
                grad_w = None
                grad_f = None
                for w in weights:
                    w.requires_grad_(True)
                N = img.shape[0]
                import copy
                import torch.autograd as autograd

                outputs = self.model(img[:pruning_batch_size].cuda())
                loss = self.Loss(outputs, target[:pruning_batch_size].cuda())
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                #TODO: Automate this ?
                outputs = self.model(img[:pruning_batch_size].cuda())
                loss = self.Loss(outputs, target[:pruning_batch_size].cuda())
                grad_f = autograd.grad(loss, weights, create_graph=True)
                z = 0
                count = 0
                for layer in self.model.modules():
                    if hasattr(layer, 'weight') and layer.weight.requires_grad == True:
                        z += (grad_w[count] * grad_f[count]).sum()
                        count += 1
                z.backward()
