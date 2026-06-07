Examples
========

This is a collection of examples in Earth2Studio that demonstrate various functionality
and commonly used workflows.

.. dropdown:: Running Examples
    :color: info
    :icon: rocket

    Earth2Studio examples can be downloaded as a notebook or runnable Python script.
    Each requires installation of different optional dependency groups or additional
    packages for the specific models used or post-processing steps.
    Use uv to auto install dependencies on execution:

    ``uv run <example_script>.py``

    If you are using a container or other type of environment, then pip installing will
    likely be needed.
    Look for the `uv inline metadata <https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>`_
    blocks of the form:

    .. code-block:: python

        # /// script
        # dependencies = [
        #   "earth2studio[viz] @ git+https://github.com/NVIDIA/earth2studio.git",
        # ]
        # ///

    Pip install these packages then execute the example with:

    ``python <example_script>.py``

    Examples that produce figures now use the consolidated ``viz`` extra. If an
    example lists model-specific extras, ``viz`` appears in the same Earth2Studio
    dependency entry so uv installs the plotting backend from the package module.

Example Script Sizes
--------------------

Line counts are tracked here to make example growth visible during review.

.. list-table::
    :header-rows: 1
    :widths: 80 20

    * - Example
      - Lines
    * - ``examples/01_getting_started/01_deterministic_workflow.py``
      - 112
    * - ``examples/01_getting_started/02_diagnostic_workflow.py``
      - 123
    * - ``examples/01_getting_started/03_ensemble_workflow.py``
      - 158
    * - ``examples/02_medium_range/01_ensemble_workflow_extend.py``
      - 198
    * - ``examples/02_medium_range/02_model_perturbation_hook.py``
      - 232
    * - ``examples/02_medium_range/03_huge_ensembles.py``
      - 199
    * - ``examples/02_medium_range/04_temporal_interpolation.py``
      - 111
    * - ``examples/02_medium_range/05_cyclone_tracking.py``
      - 203
    * - ``examples/02_medium_range/06_atlas_inference.py``
      - 174
    * - ``examples/03_downscaling/01_corrdiff_inference.py``
      - 210
    * - ``examples/03_downscaling/02_cbottle_super_resolution.py``
      - 205
    * - ``examples/03_downscaling/03_ensemble_downscaling.py``
      - 291
    * - ``examples/04_nowcasting/01_stormcast_example.py``
      - 121
    * - ``examples/04_nowcasting/02_stormcast_ensemble_example.py``
      - 221
    * - ``examples/04_nowcasting/03_stormscope_goes_example.py``
      - 245
    * - ``examples/05_data_assimilation/01_stormcast_sda.py``
      - 333
    * - ``examples/05_data_assimilation/02_healda.py``
      - 244
    * - ``examples/06_seasonal/01_seasonal_statistics.py``
      - 307
    * - ``examples/06_seasonal/02_dlesym_example.py``
      - 221
    * - ``examples/07_misc/01_distributed_manager.py``
      - 183
    * - ``examples/07_misc/02_cbottle_generation.py``
      - 268
    * - ``examples/07_misc/03_io_performance.py``
      - 419
    * - ``examples/07_misc/04_local_datasource.py``
      - 148
    * - ``examples/07_misc/05_cbottle_tc_guidance.py``
      - 177
    * - ``examples/08_extend/01_custom_prognostic.py``
      - 277
    * - ``examples/08_extend/02_custom_diagnostic.py``
      - 217
    * - ``examples/08_extend/03_custom_datasource.py``
      - 261
