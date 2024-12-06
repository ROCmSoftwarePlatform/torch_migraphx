Environment Variables
=====================

Logging
---------------

Accepted values for these variables are logging levels defined by the logging module,
refer to: https://docs.python.org/3/library/logging.html#logging-levels

NOTSET

DEBUG

INFO

WARNING

ERROR

CRITICAL


By default, torch_migraphx logs levels higher than WARNING (inclusive)

.. envvar:: TORCH_MIGRAPHX_LOGLEVEL

Default log level for all purposes. 
Default behavior is WARNING


.. envvar:: TORCH_MIGRAPHX_LOG_INTERPRETER

Log level for interpreter.

INFO outputs:

 - PyTorch graph passed to MGXInterpreter
 - Parsed MIGraphX program

DEBUG outputs:

 - Node info for each node in pytorch graph


.. envvar:: TORCH_MIGRAPHX_LOG_FX_LOWER

Log level for fx lowering.

INFO outputs:

 - Name of each subgraph that is being lowered
 - Node support summary (ie. supported, unsupported nodes)

DEBUG outputs:

 - Input shapes and pytorch graph
 - Parsed MIGraphX program that is to be compiled
 - Compiled MIGraphX program


.. envvar:: TORCH_MIGRAPHX_LOG_DYNAMO_LOWER

Log level for dynamo lowering.

INFO outputs:

 - Name of each subgraph that is being lowered
 - Input shapes and pytorch graph

DEBUG outputs:

 - Parsed MIGraphX program that is to be compiled
 - Compiled MIGraphX program


.. envvar:: TORCH_MIGRAPHX_LOG_DYNAMO_PASSES

Log level for dynamo pre lowering dynamo passes

INFO outputs:

 - Graph info before and after pre-partitioning, partitioning and post-partitioning passes

DEBUG outputs:

 - Graph info for each sub pass


.. envvar:: TORCH_MIGRAPHX_LOG_PARTITIONER

Log level for partitioner pass specifically 