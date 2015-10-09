#!/bin/bash
mkdir $HOME/spyking-circus/ 2>/dev/null
# Uses the -n option do not overwrite existing files -- could also notifiy the
# user or make backup files
cp -n -r $PREFIX/data/spyking-circus/* $HOME/spyking-circus
echo "##########################################################################" >> $PREFIX/.messages.txt
echo "# Mapping files and parameter template have been copied to $HOME/spyking-circus" >> $PREFIX/.messages.txt
echo "# To get support for faster filtering using the GPU, install 'cudamat' with the following command:" >> $PREFIX/.messages.txt
echo "#     pip install git+git://github.com/cudamat/cudamat ">> $PREFIX/.messages.txt
echo "##########################################################################" >> $PREFIX/.messages.txt
