add_executable(embedding-vocab embedding_vocab.cpp)
target_link_libraries(embedding-vocab meta-analyzers meta-util meta-io)

add_executable(embedding-cooccur embedding_cooccur.cpp)
target_link_libraries(embedding-cooccur meta-embeddings)

add_executable(glove glove.cpp)
target_link_libraries(glove meta-util
                            meta-io
                            cpptoml
                            ${CMAKE_THREAD_LIBS_INIT})

add_executable(interactive-embeddings interactive_embeddings.cpp)
target_link_libraries(interactive-embeddings meta-embeddings)

add_executable(meta-to-glove meta_to_glove.cpp)
target_link_libraries(meta-to-glove meta-util
                                    meta-io
                                    cpptoml)

add_executable(word2vec word2vec.cpp)
target_link_libraries(word2vec meta-embeddings)
