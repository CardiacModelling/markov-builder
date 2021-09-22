class settings:
    def __init__(self):
        '''Fixed parameter settings used for simulations'''
        # Relative and absolute tolerances to solve the system with, [rtol,
        # atol]
        self.solver_tolerances = [1e-5, 1e-7]

        # The value that the membrane potential is clamped too before the
        # protocol is applied (mV)
        self.holding_potential = -80
