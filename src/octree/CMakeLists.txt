project(octree)

cuda_add_library(octree ${GPU_LIBRARY_TYPE} morton.cu OPTIONS ${CUDAOPTIONS})
