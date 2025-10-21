from typing import Protocol
from . import rview

class IsmdbRview(rview.Rview, Protocol):

    def vparameters(self, filters, include_meta = True):
        pass

    def dataset(self, id, select = []):
        pass

