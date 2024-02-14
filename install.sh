#!/bin/bash 

# check libraries
echo -e "\nChecking libraries.."
python ./checks/lib_check.py

# compile CUDA shared lib
echo -e "Compiling CUDA code.."
nvcc ./cuda/gpu_aln_common.cu ./cuda/gpu_aln_noref.cu -o ./cuda/gpu_aln_pack.so -shared -Xcompiler -fPIC -lcufft -lcublas -std=c++11

# do the console things
echo -e "Configuring files for your system.."

# copy binaries from backup to where they are used (to preserve original files)
cp bin/bck/sp_isac2_gpu.py    bin/
cp bin/bck/sp_isac_applications.py bin/
cp bin/bck/sp_isac_alignment.py    bin/

# tell applications.py where to find the CUDA shared lib
sed -i.bkp "s|\"..\", \"..\", \"..\", \"cuda\"|\"$(pwd)/cuda/\"|g" ./bin/sp_isac_applications.py

# tell gpu isac to use the system's python installation and where to find the CUDA share lib
sed -i.bkp "s|REPLACE_WITH_PROPER_SHEBANG_LINE|$(dirname $(which sphire))/python|g" ./bin/sp_isac2_gpu.py
sed -i.bkp "s|\"..\", \"..\", \"..\", \"cuda\"|\"$(pwd)/cuda/\"|g" ./bin/sp_isac2_gpu.py

# link files with GUI
echo -e "Linking GPU ISAC with the SPHIRE GUI.."
ln -rsf bin/sp_isac_applications.py $(dirname $(which sphire))/sp_isac_applications.py
ln -rsf bin/sp_isac_alignment.py    $(dirname $(which sphire))/sp_isac_alignment.py
ln -rsf bin/sp_isac2_gpu.py         $(dirname $(which sphire))/sp_isac2_gpu.py

# set permissions
chmod -x bin/sp_isac_applications.py
chmod -x bin/sp_isac_alignment.py
chmod +x bin/sp_isac2_gpu.py

echo -e "\nGPU ISAC installation complete! Good luck with your project :)\n"
