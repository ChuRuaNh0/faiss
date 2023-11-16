import os
import numpy as np
import faiss

# os.makedirs("./home/hungpham/VisualSearch/data/index-repo/visualsearch_v2.1/regular/index_ondisk/14654/vn/whole_image/non_segment")

if __name__ == '__main__':
    # with open("/FAISS/faiss/visualsearch_plus_ver2.vn-1690343099.4993706.bin", mode='rb') as file: # b is important -> binary
    #     fileContent = file.read()
    # print(fileContent)
    model = faiss.read_index('/FAISS/faiss/visualsearchv1_plus.vn.onram-1679114872.177069.bin')
    vector = np.ones((1, 64), dtype = np.float32)*0.1
    vector /= np.linalg.norm(vector)
    faiss.ParameterSpace().set_index_parameter(model, "nprobe", 64)
    dis, label = model.search(x=vector, k=3)
    print(dis, label)

    _, dis, label = model.boundary_search(x=vector, lower=1.3, upper=1.33)
    print(label, dis)
    
    # dis, label = model.boundary_search_v1(x=vector, k=30, lower=0, upper=2)
    print(model.__dict__)
    dis, label = model.boundary_search_v1(x=vector, k=30, lower=0, upper=2)
    print(label, dis)