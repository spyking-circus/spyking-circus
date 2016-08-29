#!/bin/bash
mkdir $HOME/spyking-circus/ 2>/dev/null
# Uses the -n option do not overwrite existing files -- could also notifiy the
# user or make backup files
cp -n -r $PREFIX/data/spyking-circus/* $HOME/spyking-circus
echo "#############################################################################################" >> $PREFIX/.messages.txt
echo "# Mapping files and parameter template have been copied to $HOME/spyking-circus" >> $PREFIX/.messages.txt
echo "# To get support for the GPU, install 'cudamat' with the following command:" >> $PREFIX/.messages.txt
echo "#   pip install https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus ">> $PREFIX/.messages.txt
echo "############################################################################################" >> $PREFIX/.messages.txt
