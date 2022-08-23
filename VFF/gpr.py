# Copyright 2021 ST John
# Copyright 2016 James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function, absolute_import
from dataclasses import dataclass
import numpy as np
import gpflow
import tensorflow as tf
from functools import reduce
from .spectral_covariance import make_Kuu, make_Kuf, make_Kuf_np
from .kronecker_ops import kron, make_kvs_np, make_kvs
from .matrix_structures import BlockDiagMat_many
from gpflow import default_float


class GPR_1d(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernel):

        self.X, self.Y = data
        assert self.X.shape[1] == 1
        assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        mean_function = gpflow.mean_functions.Zero()
        num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = data
        self.a = a
        self.b = b
        self.ms = ms

        # Precompute static quantities
        assert np.all(self.X > a)
        assert np.all(self.X < b)
        Kuf = make_Kuf_np(self.X, a, b, ms)
        self.KufY = np.dot(Kuf, self.Y)
        self.KufKfu = np.dot(Kuf, Kuf.T)
        self.tr_YTY = np.sum(np.square(self.Y))

    def maximum_log_likelihood_objective(self):
        return tf.reduce_sum(self.elbo())

    def elbo(self):
        """ Provides a variational bound (ELBO) on the log marginal likelihood of the model """
        K_diag = self.kernel.K_diag(self.X)
        Kuu = make_Kuu(self.kernel, self.a, self.b, self.ms)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), default_float())
        D = tf.cast(tf.shape(self.Y)[1], default_float())
        
        elbo = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        elbo -= 0.5 * D * log_det_P
        elbo += 0.5 * D * Kuu.logdet()
        elbo -= 0.5 * self.tr_YTY / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo -= 0.5 * tf.reduce_sum(K_diag) / sigma2
        elbo += 0.5 * Kuu.trace_KiX(self.KufKfu) / sigma2,
        
        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov

        Kuu = make_Kuu(self.kernel, self.a, self.b, self.ms)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kus = make_Kuf(self.kernel, Xnew, self.a, self.b, self.ms)
        tmp = tf.linalg.triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.matmul(tf.transpose(tmp), tmp)
                - tf.matmul(tf.transpose(KiKus), Kus)
            )
            shape = tf.stack([1, 1, tf.shape(self.y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kernel.K_diag(Xnew)
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var
        

class GPR_additive(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernel_list):

        X, Y = data

        assert Y.shape[1] == 1
        assert X.shape[1] == len(kernel_list)
        assert a.size == len(kernel_list)
        assert b.size == len(kernel_list)
        for kernel in kernel_list:
            assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        mean_function = gpflow.mean_functions.Zero()
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = data
        self.X, self.Y = data
        self.a = a
        self.b = b
        self.ms = ms
        self.input_dim = X.shape[1]
        self.kernels = kernel_list

        # pre compute static quantities: chunk data to save memory
        self.tr_YTY = np.sum(np.square(Y))
        Mtotal = (2 * self.ms.size - 1) * X.shape[1]
        self.KufY = np.zeros((Mtotal, 1))
        self.KufKfu = np.zeros((Mtotal, Mtotal))
        for i in range(0, (X.shape[0]), 10000):
            Xchunk = X[i : i + 10000]
            Ychunk = Y[i : i + 10000]
            Kuf_chunk = np.empty((0, Xchunk.shape[0]))
            KufY_chunk = np.empty((0, Ychunk.shape[1]))
            for i, (ai, bi) in enumerate(zip(self.a, self.b)):
                assert np.all(Xchunk[:, i] > ai)
                assert np.all(Xchunk[:, i] < bi)
                Kuf = make_Kuf_np(Xchunk[:, i : i + 1], ai, bi, self.ms)
                KufY_chunk = np.vstack((KufY_chunk, np.dot(Kuf, Ychunk)))
                Kuf_chunk = np.vstack((Kuf_chunk, Kuf))
            self.KufKfu += np.dot(Kuf_chunk, Kuf_chunk.T)
            self.KufY += KufY_chunk
        self.KufY = self.KufY
        self.KufKfu = self.KufKfu

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        total_variance = reduce(tf.add, [k.variance for k in self.kernels])
        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kernels, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        # compute log marginal bound
        ND = tf.cast(num_data * output_dim, default_float())
        D = tf.cast(output_dim, default_float())

        elbo = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        elbo += -0.5 * D * log_det_P
        elbo += 0.5 * D * Kuu.logdet()
        elbo += -0.5 * self.tr_YTY / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo += -0.5 * ND * total_variance / sigma2
        elbo += 0.5 * D * Kuu.trace_KiX(self.KufKfu) / sigma2

        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov

        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kernels, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kus = tf.concat(
            [
                make_Kuf(k, Xnew[:, i : i + 1], a, b, self.ms)
                for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
            ],
            axis=0,
        )
        tmp = tf.linalg.triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = reduce(tf.add, [k.K(Xnew[:, i : i + 1]) for i, k in enumerate(self.kernels)])
            var += tf.matmul(tf.transpose(tmp), tmp)
            var -= tf.matmul(tf.transpose(KiKus), Kus)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = reduce(tf.add, [k.K_diag(Xnew[:, i : i + 1]) for i, k in enumerate(self.kernels)])
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var


class GPR_kron(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernel_list):

        for kernel in kernel_list:
            assert isinstance(kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52))

        likelihood = gpflow.likelihoods.Gaussian()
        mean_function = gpflow.mean_functions.Zero()
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = data
        self.X, self.Y = data
        self.a = a
        self.b = b
        self.ms = ms
        self.input_dim = self.X.shape[1]
        self.kernels = kernel_list

        # count the inducing variables:
        self.Ms = []
        for kern in kernel_list:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            self.Ms.append(Ncos_d + Nsin_d)

        # pre compute static quantities
        assert np.all(self.X > a)
        assert np.all(self.X < b)
        Kuf = [
            make_Kuf_np(self.X[:, i : i + 1], a, b, self.ms)
            for i, (a, b) in enumerate(zip(self.a, self.b))
        ]
        self.Kuf = make_kvs_np(Kuf)
        self.KufY = np.dot(self.Kuf, self.Y)
        self.KufKfu = np.dot(self.Kuf, self.Kuf.T)
        self.tr_YTY = np.sum(np.square(self.Y))

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        Kdiag = reduce(
            tf.multiply, [k.K_diag(self.X[:, i : i + 1]) for i, k in enumerate(self.kernels)]
        )
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kernels, self.a, self.b)]
        Kuu_solid = kron([Kuu_d.get() for Kuu_d in Kuu])
        Kuu_inv_solid = kron([Kuu_d.inv().get() for Kuu_d in Kuu])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu_solid
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kuu_logdets = [K.logdet() for K in Kuu]
        N_others = [float(np.prod(self.Ms)) / M for M in self.Ms]
        Kuu_logdet = reduce(tf.add, [N * logdet for N, logdet in zip(N_others, Kuu_logdets)])

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), default_float())
        D = tf.cast(tf.shape(self.Y)[1], default_float())
        
        elbo = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        elbo -= 0.5 * D * log_det_P
        elbo += 0.5 * D * Kuu_logdet
        elbo -= 0.5 * self.tr_YTY / sigma2
        elbo += 0.5 * tf.reduce_sum(tf.square(c))
        elbo -= 0.5 * tf.reduce_sum(Kdiag) / sigma2
        elbo += 0.5 * tf.reduce_sum(Kuu_inv_solid * self.KufKfu) / sigma2
        
        return elbo

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kernels, self.a, self.b)]
        Kuu_solid = kron([Kuu_d.get() for Kuu_d in Kuu])
        Kuu_inv_solid = kron([Kuu_d.inv().get() for Kuu_d in Kuu])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu_solid
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        Kus = [
            make_Kuf(k, Xnew[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
        ]
        Kus = tf.transpose(make_kvs([tf.transpose(Kus_d) for Kus_d in Kus]))
        tmp = tf.linalg.triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = tf.matmul(Kuu_inv_solid, Kus)
        if full_cov:
            raise NotImplementedError
        else:
            var = reduce(
                tf.multiply, [k.K_diag(Xnew[:, i : i + 1]) for i, k in enumerate(self.kernels)]
            )
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var