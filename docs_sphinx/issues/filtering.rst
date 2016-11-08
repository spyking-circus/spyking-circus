Filtering
=========

The filtering is performed once, on the data, without any copy. This has pros and cons. The pros is that this allow the code to be faster, avoiding filtering on-the-fly the data each time temporal chunks are loaded. The cons is that the user has to be careful about how this filtering is done. 

Wrong parameters
----------------

If you filled the parameter files with incorrect values either for the data type, header, or even the number of channels (i.e. with a wrong probe file), then the filtering is likely to output wrong data in the file itself. If you are facing issues with the code, always be sure that the informations displayed by the algorithm before any operations are correct, and that the data are correctly read. To be sure, use the preview GUI before launching the whole algorithm (see :doc:`Python GUI <../GUI/python>`)::

    >> spyking-circus mydata.extension -p


Interruption of the filtering
-----------------------------

The filtering is performed in parallel by several nodes, each of them in charge of a subset of all the temporal chunks. This means that if any of them is failing because of a crash, or if the filtering is interupted by any means, then you have to copy again the entiere raw file and start again. Otherwise, you are likely to filter twice some subparts of the data, leading to wrong results

Flag filter_done
----------------

To let the code know that the filtering has been performed, you can notice at the bottom of the configuration file a flag ``filter_done`` that is False by default, but that becomes ``True`` only after the filtering has been performed. As long as this parameter files is ketp along with your data, the algorithm, if relaunched, will not refilter the file. 

.. warning::

    If you delete the configuration file, but want to keep the same filtered data, then think about setting this flag manually to ``True`` 
