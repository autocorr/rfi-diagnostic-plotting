Process RFI
===========
Process and create VLA RFI plots for X and Ku Band tests. This module is to be
run under CASA v6 and Python 3.


Getting started
---------------
First, clone or download this repository, copy the ``process.cfg`` file into
the directory where CASA is to be run, and edit the paths accordingly.  Please
ensure that the ``SDM_DIR`` is set to the directory where the ASDM files are
stored. Then start CASA and run:

.. code-block:: python

   import sys
   sys.path.append("/<PATH>/<TO>/rfi-diagnostic-plotting")
   from process_rfi.core import process_all_executions
   process_all_executions(overwrite=False)


License
-------
The pipeline is authored by Brian Svoboda, Paul Demorest, and Urvashi Rao. The
code and documentation is released under the MIT License. A copy of the license
is supplied in the LICENSE file.


