Formatting your data
====================

Input format for the code
-------------------------

Currently, the code only accepts plain binary data. To be more precise, suppose you have *N* channels 

.. math::

   c_0, c_1, ... , c_N

And if you assume that :math:`c_i(t)` is the value of channel :math:`c_i` at time *t*, then your datafile should be a giant raw file with values

.. math::

   c_0(0), c_1(0), ... , c_N(0), c_0(1), ..., c_N(1), ... c_N(T)


This is simply the flatten version of your recordings matrix, with size *N* x *T* 

.. note::

    The values can be saved in your own format (``int16``, ``uint16``, ``int8``, ``float32``). You simply need to specify that to the code


Header
------

Your datafile can have a header, with some information about your recordings before the raw data themselves. Quite often, this header has a variable size, so either you know it (see ``data_offset``), either your recording has been exported from a MultiChannel Device (using MCDataTools), and therefore, by setting ``data_offset = MCS``, the size is automatically handled. If you do not have any header, just set ``data_offset = 0``


.. note ::
    
    The best way to see if your data are properly loaded is to use the preview mode of the code (see :doc:`documentation on Python GUI <../GUI/python>`). If you have issues, please be in touch with pierre.yger@inserm.fr


Future plans
------------

Hopefully in a near future, we plan to enhance the interface between SpyKING CIRCUS and various file formats. Most likely using Neo, we should be able to read/write the data without a need for a proper raw file.


