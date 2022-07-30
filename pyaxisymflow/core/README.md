Build instructions
------------------

You need a working `c++` compiler to create the python bindings. Make sure to
install the python (see `requirements.txt`) and system prerequisities
(`pybind11` headers, `make`). Run:

```sh
cd src && make bind
```
which generates the `pybind` binding files from the `c++` sources, which are
then compiled as shared libraries. These files can then be found in the `core`
directory and can be called in `python` as:

```python
from core.XXX import YYY
```
