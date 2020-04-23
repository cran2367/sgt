library(matrixStats)
library(dplyr)
source('sgt.R', echo = F)
source('kmeans.R', echo = F)

######################################################################
######## Validate SGT output with a simple sequence example ##########
######################################################################

alphabet_set      <- c("A", "B", "C")
alphabet_set_size <- length(alphabet_set)

seq <- "BBACACAABA"

kappa <- 5

###### Algorithm 1 ######
sgt_parts_alg1 <- f_sgt_parts(sequence = seq, kappa = kappa, alphabet_set_size = alphabet_set_size)
print(sgt_parts_alg1)

sgt <- f_SGT(W_kappa = sgt_parts_alg1$W_kappa, W0 = sgt_parts_alg1$W0, 
             Len = sgt_parts_alg1$Len, kappa = kappa)  # Set Len = NULL for length-sensitive SGT.
print(sgt)

###### Algorithm 2 ######
seq_split <- f_seq_split(sequence = seq)
seq_alphabet_positions <- f_get_alphabet_positions(sequence_split = seq_split, alphabet_set = alphabet_set)

sgt_parts_alg2 <- f_sgt_parts_using_element_positions(seq_alphabet_positions = seq_alphabet_positions, 
                                                      alphabet_set = alphabet_set, 
                                                      kappa = kappa)

sgt <- f_SGT(W_kappa = sgt_parts_alg2$W_kappa, W0 = sgt_parts_alg2$W0, 
             Len = sgt_parts_alg2$Len, kappa = kappa)  # Set Len = NULL for length-sensitive SGT.


############################################################################
######## Demo: Performing a Clustering operation on a seq dataset ##########
############################################################################

## The dataset contains all roman letters, A-Z.
dataset <- read.csv("../data/simulated-sequence-dataset.csv", header = T, stringsAsFactors = F)

sgt_parts_sequences_in_dataset <- f_SGT_for_each_sequence_in_dataset(sequence_dataset = dataset, 
                                                                     kappa = 5, alphabet_set = LETTERS, 
                                                                     spp = NULL, sgt_using_alphabet_positions = T)
  
  
input_data <- f_create_input_kmeans(all_seq_sgt_parts = sgt_parts_sequences_in_dataset, 
                                    length_normalize = T, 
                                    alphabet_set_size = 26, 
                                    kappa = 5, trace = TRUE, 
                                    inv.powered = T)
K = 5
clustering_output <- f_kmeans(input_data = input_data, K = K, alphabet_set_size = 26, trace = T)

cc <- f_clustering_accuracy(actual = c(strtoi(dataset[,1])), pred = c(clustering_output$class), K = K, type = "f1")  
print(cc)

######## Clustering on Principal Components of SGT features ########
num_pcs <- 5  # Number of principal components we want
input_data_pcs <- f_pcs(input_data = input_data, PCs = num_pcs)$input_data_pcs

clustering_output_pcs <- f_kmeans(input_data = input_data_pcs, K = K, alphabet_set_size = sqrt(num_pcs), trace = F)

cc <- f_clustering_accuracy(actual = c(strtoi(dataset[,1])), pred = c(clustering_output_pcs$class), K = K, type = "f1")  
print(cc)
