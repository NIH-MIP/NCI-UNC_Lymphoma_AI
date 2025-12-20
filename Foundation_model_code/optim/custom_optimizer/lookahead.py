import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.

    Wraps another optimizer and performs lookahead updates every k steps:
      slow = slow + alpha * (fast - slow)
      fast = slow

    Args:
        optimizer (Optimizer): Inner (fast) optimizer.
        k (int): Number of fast steps per lookahead update.
        alpha (float): Interpolation factor (0 < alpha <= 1).
        pullback_momentum (str): 'none', 'reset', or 'pullback'.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = "none",
    ):
        # we won't call super().__init__ because we're wrapping another Optimizer
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum
        self.step_counter = 0

        # delegate param_groups and defaults
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults

        # initialize slow/backup weights and momentum
        for group in self.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['slow_param'] = p.data.clone().detach()
                state['backup_param'] = p.data.clone().detach()
                mom = state.get('momentum_buffer')
                state['slow_mom'] = mom.clone().detach() if mom is not None else torch.zeros_like(p.data)

    def step(self, closure=None):
        """
        Take a step with the inner optimizer, then every k steps perform the lookahead update.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # fast step
        loss = self.optimizer.step(closure)

        self.step_counter += 1
        if self.step_counter % self.k != 0:
            return loss

        # lookahead update: mix slow + fast, then sync fast=slow
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                slow = state['slow_param']
                fast = p.data

                # backup fast for momentum pullback
                state['backup_param'] = fast.clone().detach()

                # slow <- slow + alpha * (fast - slow)
                updated = slow + self.alpha * (fast - slow)
                p.data.copy_(updated)
                state['slow_param'] = updated.clone().detach()

                # momentum pullback
                if self.pullback_momentum == "pullback":
                    mom = state.get('momentum_buffer')
                    if mom is not None:
                        slow_mom = state['slow_mom']
                        new_mom = slow_mom + self.alpha * (mom - slow_mom)
                        self.optimizer.state[p]['momentum_buffer'] = new_mom.clone().detach()
                        state['slow_mom'] = new_mom.clone().detach()
                elif self.pullback_momentum == "reset":
                    if 'momentum_buffer' in self.optimizer.state[p]:
                        self.optimizer.state[p]['momentum_buffer'] = state['slow_mom'].clone().detach()

        return loss

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        """
        Returns a dict containing the inner optimizer state + lookahead buffers.
        """
        return {
            'inner': self.optimizer.state_dict(),
            'slow_buffers': {
                id(p): {
                    'slow_param': state['slow_param'],
                    'slow_mom': state['slow_mom'],
                }
                for group in self.param_groups
                for p in group['params']
                for state in [self.optimizer.state[p]]
            },
            'step_counter': self.step_counter,
            'alpha': self.alpha,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the inner optimizer state and restores lookahead buffers.
        """
        self.optimizer.load_state_dict(state_dict['inner'])
        slow = state_dict['slow_buffers']
        for group in self.param_groups:
            for p in group['params']:
                buf = slow.get(id(p))
                if buf is not None:
                    st = self.optimizer.state[p]
                    st['slow_param'] = buf['slow_param'].clone().detach()
                    st['slow_mom'] = buf['slow_mom'].clone().detach()
        self.step_counter = state_dict['step_counter']
        self.alpha = state_dict['alpha']
        self.k = state_dict['k']
        self.pullback_momentum = state_dict['pullback_momentum']

    def __getattr__(self, name):
        # Delegate any missing attributes to the inner optimizer
        if name in ('optimizer', 'param_groups', 'defaults'):
            return object.__getattribute__(self, name)
        return getattr(self.optimizer, name)
