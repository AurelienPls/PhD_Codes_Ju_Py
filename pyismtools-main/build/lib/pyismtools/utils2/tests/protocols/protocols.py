import typing

import pyismtools.utils2.domain.model as m
import pyismtools.utils2.domain.protocols as mproto

import pyismtools.utils2.tests.protocols.prout_protocol as pp

def main():

    proutm = m.define('prout')

    prout = proutm({})

    # define prout_protocol implementation
    def _pprout(a: str):
        print('prout: ', a)

    mproto.extend(prout, 'prout', {'pprout': _pprout})

    pp.pprout(prout, 3)


main()


