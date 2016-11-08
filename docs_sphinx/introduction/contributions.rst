Future plans and contributions
==============================

Here is a non-exhaustive list of the features that we are currently working on, and that should make it into future releases of the software

Real Time spike sorting
-----------------------

This is the most challenging task, and we are thinking about what is the best way to properly implement it. Such a real-time spike sorting for dense arrays is within reach, but several challenges need to be addressed to make it possible. Data will be read from memory streams, and templates will be updated on-the-fly. The plan is to have spatio-temporal templates tracking cells over time, at a cost of a small temporal lag that can not be avoided because of the template-matching step.

Better, faster, stronger
------------------------

GPU kernels should be optimized to increase the speed of the algorithm, and we are always seeking for optimizations along the road. For Real-Time spike sorting, if we want it to be accurate for thousands of channels, any optimizations is welcome. 


Contributions
-------------

If you have ideas, or if you want to contribute to the software, with the same idea that we should develop a proper and unified framework for semi-automated spike sorting, please do not hesitate to contact pierre.yger@inserm.fr . Currently, the code itself is not properly documented, as our main focus was to first get a stable working algorithm. Now that this goal is now achieved, we can dive more into software development and enhance its modularity.