# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:11:00 2020
Pour installer XSteam, dans la console anaconda :
    pip install pyXSteam
"""

import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

p = np.linspace(1,50,300)
Tsat = np.asarray([steamTable.tsat_p(k) for k in p])
plt.figure()
plt.plot(p,Tsat)
