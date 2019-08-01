import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.visual import plot_alignment


class TacotronVisualsTest(unittest.TestCase):

    @unittest.skip("Manual alignment plot test")
    def test_alignment_plot(self):
        import matplotlib
        matplotlib.use("TkAgg")
        straight_alignment = np.identity(100)
        straight_alignment = np.pad(straight_alignment, ((0, 20), (0, 0)), 'constant')
        real_text_len = 60
        real_spec_len = 90
        fig = plot_alignment(straight_alignment, text_len=real_text_len,
                             spec_len=real_spec_len)
        plt.show(fig)
