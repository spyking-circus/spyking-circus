Whitening
=========

No silences are detected
------------------------

This section should be pretty robust, and the only error that you could get is a message saying that no silence were detected. If this is the case, this is likely that the parameters are wrong, and that the data are not properly understood. Be sure that your data are properly loaded by using the preview mode::

	>> spyking-circus mydata.extension -p


If this is the case, please try to reduce the ``safety_time`` value. If no silences are detected, then your data may not be properly loaded.

Whitening is disabled because of NaNs
-------------------------------------

Again, this should be rare, and if this warning happens, you may try to get rid of this warning by changing the parameters of the ``whitening`` section. Try for example to increase ``safety_time`` for example to ``3``, or try to change the value of ``chunk_size``. We may enhance the robustness of the whitening in future releases.