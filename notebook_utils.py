import os
import re
import pickle
import sympy

import numpy as np
import scipy as sp

import plotly
import plotly.offline        as py
import plotly.graph_objs     as go
import plotly.figure_factory as ff

from plotly.offline  import download_plotlyjs, init_notebook_mode
from IPython.display import display, display_markdown, Image

sympy.init_printing()
init_notebook_mode(connected=True)
show = lambda s: display_markdown(s, raw=True)
