# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Selected definitions from torchdyn 1.0.6; see ../SOURCE.json.
# Loaded by AST to avoid unrelated optional package dependencies.

def construct_dopri5(dtype):
    c = torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=dtype)
    a = [
        torch.tensor([1 / 5], dtype=dtype),
        torch.tensor([3 / 40, 9 / 40], dtype=dtype),
        torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=dtype),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype),
    ]
    bsol = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=dtype)
    berr = torch.tensor([1951 / 21600, 0, 22642 / 50085, 451 / 720, -12231 / 42400, 649 / 6300, 1 / 60.], dtype=dtype)

    dmid = torch.tensor([-1.1270175653862835, 0., 2.675424484351598, -5.685526961588504, 3.5219323679207912,
                         -1.7672812570757455, 2.382468931778144])
    return (c, a, bsol, bsol - berr)
