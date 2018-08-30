"""
Given `trans_ab = _BidirectionalTransformationImp("a", "b", a2b, lambda x: x.inv())`, we have by default:
`trans_ab.a_to_b` which is `a2b` and `trans_ab.b_to_a` which is `a2b.inv()`.
"""
import operator
from collections import namedtuple

import numpy as np


class _BidirectionalTransformationImp(object):

    def __init__(self, _from, _to, trans, inverse_func, apply_func=None):
        if '_to_' in _from or '_to_' in _to:
            raise RuntimeError("_to_ should be in name frame name.")
        self._from = _from
        self._to = _to
        self._trans = trans
        self._f_invese = inverse_func
        self._apply = apply_func

    def __getattr__(self, key):
        candidate_trans_dir = ['{}_to_{}'.format(s, t) for (s, t) in [(self._from, self._to), (self._to, self._from)]]

        if key not in candidate_trans_dir:
            raise RuntimeError("transformation not supported, from/to in {}/{}, but {} provided".format(
                self._from, self._to, key))

        if key == candidate_trans_dir[0]:
            return self._trans

        return self._f_invese(self._trans)

    def __check(self, operand):
        assert self._apply(self._f_invese(self._trans), self._apply(self._trans, operand)) == operand

    def inv(self):
        # It's user' responsibility to make sure f(f_inv) == I. => _to = f_inv(_from) <==> f(_to) == _from
        # which can be verified using __check.
        return _BidirectionalTransformationImp(self._to, self._from, self._trans, self._f_invese, self._apply)

    def apply(self, x):
        assert self._apply
        return self._apply(self._trans, x)

    def apply_inv(self, x, op):
        assert self._apply
        return self._apply(self._f_invese(self._trans), x)


class BiScalarTrans(_BidirectionalTransformationImp):

    @staticmethod
    def _inv(x):
        return 1.0 / x

    def __init__(self, _from, _to, scale):
        super(BiScalarTrans, self).__init__(_from, _to, scale, BiScalarTrans._inv, operator.__mul__)


class BiRotationTrans(_BidirectionalTransformationImp):

    @staticmethod
    def _inv(x):
        return x.T

    def __init__(self, _from, _to, rmat):
        super(BiRotationTrans, self).__init__(_from, _to, rmat, BiRotationTrans._inv, np.matmul)


class BiRigidTrans(_BidirectionalTransformationImp):

    @staticmethod
    def _inv(rigid_trans):
        return rigid_trans.inv()

    def __init__(self, _from, _to, rigid_trans):
        super(BiRigidTrans, self).__init__(_from, _to, rigid_trans, BiRigidTrans._inv)


class BiAffineTrans(_BidirectionalTransformationImp):

    @staticmethod
    def _inv(x):
        return rigid.inv_rt_affine(x)

    def __init__(self, _from, _to, affine_mat):
        super(BiAffineTrans, self).__init__(_from, _to, affine_mat, BiAffineTrans._inv)
