# Project of Parallel Computing

Álvaro Freixo - 93116

Rita Ferrolho - 88822

```bash
###################
## generic files ##
###################

# compile openMP, CUDA global and CUDA shared. CUDA files have to be compiled and executed on banana server. 
make

# execute openMP file.
./harrisDetectorOpenMP

# execute CUDA global. Needs to be executed on banana server.
./harrisDetectorCuda

# execute CUDA shared. Needs to be executed on banana server.
./harrisDetectorCudaShared

# compare images
./testDiffs -i imgInput -o imgOutput -r imgReference


################
## test files ##
################

# generate different OpenMP .c files, compile them, and compare generated images.
python test_generateOpenMPfiles.py

# calculate efficiency of each OpenMP test, store them in a table.
python resultsOpenMP.py

# generate different CUDA global files, compile them, and compare generated images. Needs to be executed on banana server.
python tests_generateCudaScripts.py

# calculate efficiency of each CUDA global test, store them in a table.
python resultsCuda.py

# generate different CUDA shared files, compile them, and compare generated images. Needs to be executed on banana server.
python tests_generateCudaSharedScripts.py

# calculate efficiency of each CUDA shared test, store them in a table.
python resultsCudaShared.py


#####################
## banana commands ##
#####################

# access banana server 
ssh cp0102@banana.ua.pt

# transfer files or directories from local to banana
scp [-r] 'local/path' cp0102@banana.ua.pt:'./server/path'

# transfer files or directories from banana to local
scp [-r] cp0102@banana.ua.pt:/server/path './local/path'
```