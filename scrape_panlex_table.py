import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize



URL = "https://vocab.panlex.org/tgl-000/eng-000?page="

table_MN = pd.read_html(URL)