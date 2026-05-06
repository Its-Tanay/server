#!/usr/bin/python
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

sys.path.append("../common")

import unittest
import torch

import test_util as tu
import tritonclient.http as http


class TorchAotiTest(tu.TestResultCollector):

    def _get_complex_input_shape(self):
        return (1, 16)

    def _get_complex_output_shape(self):
        return (1, 16)

    def _get_complex_input_data(self, shape):

        return [
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
            torch.randint(low=0, high=127, size=shape, dtype=torch.int8).numpy(),
        ]

    def test_complex_named(self):

        model_name = "torch_aoti_complex_named"

        input_data = self._get_complex_input_data()

        with http.InferenceServerClient("localhost:8000") as client:
            inputs = [
                http.InferInput("ARGS[0]", input_data[0].shape, "INT8"),
                http.InferInput("ARGS[1]", input_data[1].shape, "INT8"),
                http.InferInput("ARGS[2][option1]", input_data[2].shape, "INT8"),
                http.InferInput("ARGS[2][option2]", input_data[3].shape, "INT8"),
            ]

            inputs[0].set_data_from_numpy(input_data[0], binary_data=True)
            inputs[1].set_data_from_numpy(input_data[1], binary_data=True)
            inputs[2].set_data_from_numpy(input_data[2], binary_data=True)
            inputs[3].set_data_from_numpy(input_data[3], binary_data=True)

            output_names = [
                "RESULT[AAA]",
                "RESULT[BBB][0]",
                "RESULT[BBB][1]",
                "RESULT[CCC][option1]",
                "RESULT[CCC][option2]",
                "RESULT[ZZZ]",
            ]

            outputs = []
            for output_name in output_names:
                outputs.append(
                    http.InferRequestedOutput(output_name, binary_data=True)
                )

            output_data = []
            results = client.infer(model_name, inputs, outputs=outputs)

            for output_name in output_names:
                output_data.append(results.as_numpy(output_name))

            assert len(outputs) == len(output_data)
            for data in output_data:
                assert data.shape == self._get_complex_output_shape()

            assert (output_data[0] == (input_data[0] + input_data[1])).all()
            assert (output_data[1] == input_data[0]).all()
            assert (output_data[2] == input_data[1]).all()
            assert (output_data[3] == input_data[2]).all()
            assert (output_data[4] == input_data[3]).all()
            assert (output_data[5] == (input_data[0] - input_data[1])).all()


if __name__ == "__main__":
    unittest.main()
