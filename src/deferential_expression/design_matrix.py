import pandas as pd
import numpy as np

from formulaic_contrasts import FormulaicContrasts

class DesignMixin:
    def make_design(self, formula, samples):
        fc = FormulaicContrasts(samples, formula)
        return fc
    def get_design(self, fc: FormulaicContrasts):
        return pd.DataFrame(fc.design_matrix)