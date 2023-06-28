import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import numpy as np
import torch.nn as nn


class DSNN(nn.Module):
    """
    Rate-coded DSNN where observations are converted into spike trains in the input layer
    """
    def __init__(self, architecture, beta, threshold, reset_mechanism, time_steps, seed):
        super().__init__()

        torch.manual_seed(seed)

        self.architecture = architecture
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.time_steps = time_steps

        self.lif0 = snn.Leaky(beta, threshold=threshold)
        self.fc1 = nn.Linear(architecture[0], architecture[1])
        self.lif1 = snn.Leaky(beta, threshold=threshold)
        self.fc2 = nn.Linear(architecture[1], architecture[2])
        self.lif2 = snn.Leaky(beta, threshold=threshold)
        self.fc3 = nn.Linear(architecture[2], architecture[3])
        self.lif3 = snn.Leaky(beta, threshold=threshold, reset_mechanism='none')

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        #spike_data = spikegen.rate(x, num_steps=self.time_steps)
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        mem0_rec = []
        mem1_rec = []
        mem2_rec = []
        mem3_rec = []
        spk0_rec = []
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        for step in range(self.time_steps):
            cur0 = x
            spk0, mem0 = self.lif0(cur0, mem0)
            #cur1 = self.fc1(spike_data[step])
            cur1 = self.fc1(spk0)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)


class H1RateCodedDSNN(nn.Module):
    def __init__(self, architecture, alpha, beta, threshold, reset_mechanism, time_steps, seed):
        super().__init__()
        torch.manual_seed(seed)

        self.architecture = architecture
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.time_steps = time_steps

        self.fc1 = nn.Linear(architecture[0], architecture[1], bias=False)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(architecture[1], architecture[2], bias=False)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold)
        self.fc3 = nn.Linear(architecture[2], architecture[3], bias=False)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, reset_mechanism='none')

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        with torch.no_grad():
            torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1/np.sqrt(self.architecture[0]))
            torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1/np.sqrt(self.architecture[1]))
            torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=1/np.sqrt(self.architecture[2]))

    def forward(self, x):
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        mem1_rec = []
        mem2_rec = []
        mem3_rec = []
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        for step in range(self.time_steps):
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            cur3 = self.fc3(spk2)
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
