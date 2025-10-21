from pyismtools.utils2.domain import protocols as mproto

PROTOCOL = 'prout'

def pprout(self, a: str):
    return mproto.protocolfunc(self, PROTOCOL, 'pprout', a)
