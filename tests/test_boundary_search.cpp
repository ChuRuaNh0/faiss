// // // /**
// // //  * Copyright (c) Facebook, Inc. and its affiliates.
// // //  *
// // //  * This source code is licensed under the MIT license found in the
// // //  * LICENSE file in the root directory of this source tree.
// // //  */

// #include <cinttypes>
// #include <cstdio>
// #include <cstdlib>

// #include <memory>
// #include <random>
// #include <thread>
// #include <vector>
// #include <gtest/gtest.h>
// #include <faiss/AutoTune.h>
// // // #include <faiss/IVFlib.h>
// // // #include <faiss/IndexBinaryIVF.h>
// #include <faiss/IndexIVF.h>
// // // #include <faiss/IndexPreTransform.h>
// // // #include <faiss/VectorTransform.h>
// #include <faiss/index_factory.h>
// #include <faiss/index_io.h>
// // #include <faiss/IndexIVF.h>
// #include <faiss/Index.h>
// #include <faiss/IndexFlat.h>
// #include <faiss/IndexScalarQuantizer.h>
// // #include <faiss/IndexIVFScalarQuantizer.h>
// #include <iostream>

// using namespace faiss;

// namespace {
//     // dimension of the vectors to index
//     int d = 32;
//     // nb of training vectors
//     size_t nt = 5000;
//     // size of the database points per window step
//     size_t nb = 1000;
//     // nb of queries
//     size_t nq = 200;
//     int k = 10;
//     std::mt19937 rng;

//     std::vector<float> make_data(size_t n) {
//         std::vector<float> database(n * d);
//         std::uniform_real_distribution<> distrib;
//         for (size_t i = 0; i < n * d; i++) {
//             database[i] = distrib(rng);
//         }
//         return database;
//     }

//     std::unique_ptr<Index> make_trained_index(
//         const char* index_type,
//         MetricType metric_type) 
//     {
//         auto index = std::unique_ptr<Index>(index_factory(d, index_type, metric_type));
//         auto xt = make_data(nt);
//         index->train(nt, xt.data());
//         ParameterSpace().set_index_parameter(index.get(), "nprobe", 4);
//         return index;
//     }

//     std::vector<idx_t> search_index(Index* index, const float* xq, float lower, float upper) 
//     {
//         std::vector<idx_t> I(k * nq);
//         // std::vector<float> D(k * nq);
//         // index->search(nq, xq, k, D.data(), I.data());
//         index->boundary_search(nq, xq, lower, upper, nullptr, nullptr);
//         return I;
//     }

//     // /*************************************************************
//     //  * Test functions for a given index type
//     //  *************************************************************/

//     void test_lowlevel_access(const char* index_key, MetricType metric, float lower, float upper) 
//     {
//         std::unique_ptr<Index> index = make_trained_index(index_key, metric);

//         auto xb = make_data(nb);
//         index->add(nb, xb.data());

//         for(float obj:xb){
//             std::cout<<"std: "<<obj<< std::endl;
//         }

//         auto ref_I = search_index(index.get(), xb.data(), 0.3, 0.8);

//          for(float id:ref_I){
//             std::cout<<"ID: "<<id<< std::endl;
//         }



//     } // anonymous namespace
// }

// // TEST(TestLowLevelIVF, IVFFP16) {
// //     test_lowlevel_access("IVF1024,FP16", METRIC_L2, 0.3, 0.8);
// // }


// int main() {
//     // Set dimension and number of clusters
//     int d = 64;  // dimension
//     int ncentroids = 1024;  // number of centroids (cells)
//     faiss::MetricType metric = faiss::METRIC_L2; // Euclidean distance
//     // Create a quantizer (IndexFlatL2 is used here, but you can choose another quantizer)
//     IndexFlatL2 quantizer(d);
//     // tquantize = new IndexFlatL2(quantizer);
//     // IndexFlatL2 quantizer;
//     // ScalarQuantizer::QuantizerType index_type = QT_fp16;
//     // // Create an IndexIVFScalarQuantizer with 16-bit floating-point vectors
//     // IndexIVFScalarQuantizer index(  &quantizer,
//     //                                 d,
//     //                                 ncentroids,
//     //                                 metric,
//     //                                 true
//     //                             );
//     index.quantizer_trains_alone = true;  // set to true if you want to train the quantizer separately
//     // std::unique_ptr<Index> index = make_trained_index(index_type, metric);

//     // Add vectors to the index (your data goes here)
//     int n = 10000;
//     float* xb = new float[d * n];  // your data
//     // ... populate xb ...

//     index.train(n, xb);
//     index.add(n, xb);

//     // Perform a range search (your query goes here)
//     float* xq = new float[d];  // your query
//     // ... populate xq ...

//     // Define a range parameter
//     float radius = 0.2;

//     // Output buffers
//     faiss::RangeSearchResult result(1, n);
//     // result.distances = new float[n * 5];  // Adjust the size as needed
//     // result.labels = new idx_t[n * 5];
//     // idx_t* am = result.labels;
//     // result.distances = new float[result.lims * 5];  // 'lim' is the limit on the number of entries per list
//     // result.labels = new idx_t[result.lims * 5];
    
//     // Perform the range search
//     // Index* index = index.get();
//     // for(int i=0; i<result.labels.size(); ++i){
//     //     std::cout<< am[i]<<std::endl;
//     // }
//     // index.get()->boundary_search(1, xq, 0.2, 0.5, &result);
//     index.range_search(1, xq, radius, &result);

//     // Print the results
//     for (int i = 0; i < result.lims[1]; i++) {
//         for (int j = result.lims[i]; j < result.lims[i + 1]; j++) {
//             std::cout << "Query " << i << ", result " << j - result.lims[i] << ": "
//                       << "Index = " << result.labels[j] << ", Distance = " << result.distances[j] << std::endl;
//         }
//     }

//     // Clean up memory
//     delete[] xb;
//     delete[] xq;
//     delete[] result.distances;
//     delete[] result.labels;

//     return 0;
// }

// // int main()
// // {
// //     std::unique_ptr<Index> index = make_trained_index("IVF32,SQ8", METRIC_L2);
// //     std::vector<idx_t> vector = search_index(index, )
// // }










// // // TEST(TestLowLevelIVF, PCAIVFFlatL2) {
// // //     test_lowlevel_access("PCAR16,IVF32,Flat", METRIC_L2);
// // // }

// // // TEST(TestLowLevelIVF, IVFFlatIP) {
// // //     test_lowlevel_access("IVF32,Flat", METRIC_INNER_PRODUCT);
// // // }

// // // TEST(TestLowLevelIVF, IVFSQL2) {
// // //     test_lowlevel_access("IVF32,SQ8", METRIC_L2);
// // // }

// // // TEST(TestLowLevelIVF, IVFSQIP) {
// // //     test_lowlevel_access("IVF32,SQ8", METRIC_INNER_PRODUCT);
// // // }

// // // TEST(TestLowLevelIVF, IVFPQL2) {
// // //     test_lowlevel_access("IVF32,PQ4np", METRIC_L2);
// // // }

// // // TEST(TestLowLevelIVF, IVFPQIP) {
// // //     test_lowlevel_access("IVF32,PQ4np", METRIC_INNER_PRODUCT);
// // // }

// // /*************************************************************
// //  * Same for binary (a bit simpler)
// //  *************************************************************/

// // namespace {

// // int nbit = 256;

// // // here d is used the number of ints -> d=32 means 128 bits

// // std::vector<uint8_t> make_data_binary(size_t n) {
// //     std::vector<uint8_t> database(n * nbit / 8);
// //     std::uniform_int_distribution<> distrib;
// //     for (size_t i = 0; i < n * d; i++) {
// //         database[i] = distrib(rng);
// //     }
// //     return database;
// // }

// // std::unique_ptr<IndexBinary> make_trained_index_binary(const char* index_type) {
// //     auto index = std::unique_ptr<IndexBinary>(
// //             index_binary_factory(nbit, index_type));
// //     auto xt = make_data_binary(nt);
// //     index->train(nt, xt.data());
// //     return index;
// // }



// // namespace {

// // void test_threaded_search(const char* index_key, MetricType metric) {
// //     std::unique_ptr<Index> index = make_trained_index(index_key, metric);

// //     auto xb = make_data(nb);
// //     index->add(nb, xb.data());

// //     /** handle the case if we have a preprocessor */

// //     const IndexPreTransform* index_pt =
// //             dynamic_cast<const IndexPreTransform*>(index.get());

// //     int dt = index->d;
// //     const float* xbt = xb.data();
// //     std::unique_ptr<float[]> del_xbt;

// //     if (index_pt) {
// //         dt = index_pt->index->d;
// //         xbt = index_pt->apply_chain(nb, xb.data());
// //         if (xbt != xb.data()) {
// //             del_xbt.reset((float*)xbt);
// //         }
// //     }

// //     IndexIVF* index_ivf = ivflib::extract_index_ivf(index.get());

// //     /** Test independent search
// //      *
// //      * Manually scans through inverted lists, computing distances and
// //      * ordering results organized in a heap.
// //      */

// //     // sample some example queries and get reference search results.
// //     auto xq = make_data(nq);
// //     auto ref_I = search_index(index.get(), xq.data());

// //     // handle preprocessing
// //     const float* xqt = xq.data();
// //     std::unique_ptr<float[]> del_xqt;

// //     if (index_pt) {
// //         xqt = index_pt->apply_chain(nq, xq.data());
// //         if (xqt != xq.data()) {
// //             del_xqt.reset((float*)xqt);
// //         }
// //     }

// //     // quantize the queries to get the inverted list ids to visit.
// //     int nprobe = index_ivf->nprobe;

// //     std::vector<idx_t> q_lists(nq * nprobe);
// //     std::vector<float> q_dis(nq * nprobe);

// //     index_ivf->quantizer->search(nq, xqt, nprobe, q_dis.data(), q_lists.data());

// //     // now run search in this many threads
// //     int nproc = 3;

// //     for (int i = 0; i < nq; i++) {
// //         // one result table per thread
// //         std::vector<idx_t> I(k * nproc, -1);
// //         float default_dis = metric == METRIC_L2 ? HUGE_VAL : -HUGE_VAL;
// //         std::vector<float> D(k * nproc, default_dis);

// //         auto search_function = [index_ivf,
// //                                 &I,
// //                                 &D,
// //                                 dt,
// //                                 i,
// //                                 nproc,
// //                                 xqt,
// //                                 nprobe,
// //                                 &q_dis,
// //                                 &q_lists](int rank) {
// //             const InvertedLists* il = index_ivf->invlists;

// //             // object that does the scanning and distance computations.
// //             std::unique_ptr<InvertedListScanner> scanner(
// //                     index_ivf->get_InvertedListScanner());

// //             idx_t* local_I = I.data() + rank * k;
// //             float* local_D = D.data() + rank * k;

// //             scanner->set_query(xqt + i * dt);

// //             for (int j = rank; j < nprobe; j += nproc) {
// //                 int list_no = q_lists[i * nprobe + j];
// //                 if (list_no < 0)
// //                     continue;
// //                 scanner->set_list(list_no, q_dis[i * nprobe + j]);

// //                 scanner->scan_codes(
// //                         il->list_size(list_no),
// //                         InvertedLists::ScopedCodes(il, list_no).get(),
// //                         InvertedLists::ScopedIds(il, list_no).get(),
// //                         local_D,
// //                         local_I,
// //                         k);
// //             }
// //         };

// //         // start the threads. Threads are numbered rank=0..nproc-1 (a la MPI)
// //         // thread rank takes care of inverted lists
// //         // rank, rank+nproc, rank+2*nproc,...
// //         std::vector<std::thread> threads;
// //         for (int rank = 0; rank < nproc; rank++) {
// //             threads.emplace_back(search_function, rank);
// //         }

// //         // join threads, merge heaps
// //         for (int rank = 0; rank < nproc; rank++) {
// //             threads[rank].join();
// //             if (rank == 0)
// //                 continue; // nothing to merge
// //             // merge into first result
// //             if (metric == METRIC_L2) {
// //                 maxheap_addn(
// //                         k,
// //                         D.data(),
// //                         I.data(),
// //                         D.data() + rank * k,
// //                         I.data() + rank * k,
// //                         k);
// //             } else {
// //                 minheap_addn(
// //                         k,
// //                         D.data(),
// //                         I.data(),
// //                         D.data() + rank * k,
// //                         I.data() + rank * k,
// //                         k);
// //             }
// //         }

// //         // re-order heap
// //         if (metric == METRIC_L2) {
// //             maxheap_reorder(k, D.data(), I.data());
// //         } else {
// //             minheap_reorder(k, D.data(), I.data());
// //         }

// //         // check that we have the same results as the reference search
// //         for (int j = 0; j < k; j++) {
// //             EXPECT_EQ(I[j], ref_I[i * k + j]);
// //         }
// //     }
// // }

// // } // namespace

// // TEST(TestLowLevelIVF, ThreadedSearch) {
// //     test_threaded_search("IVF32,Flat", METRIC_L2);
// // }
