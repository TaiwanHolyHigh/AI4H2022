# -*- coding: utf-8 -*-
"""bokeh互動式資料視覺化.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16dqDpV2nKeCCemQ0krpTu8xFFVUMwEAk

## 參考資料
- [Bokeh documentation](https://docs.bokeh.org/en/latest/)
- https://docs.bokeh.org/en/latest/docs/user_guide/basic/scatters.html
"""

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()

from bokeh.plotting import figure, show

p = figure(width=400, height=400)

# add a circle renderer with a size, color, and alpha
p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

# show the results
show(p)

from bokeh.plotting import Histogram, show
import numpy as np

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
hist = Histogram(normal_samples)
show(hist)