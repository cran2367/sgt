
library(caret)
library(e1071)
library(gtools)
source('libraries.R', echo = FALSE)
source('functions_dev_v3_cleaned.R', echo=TRUE)
source('functions_markov_clustering.R', echo=TRUE)
source('functions_kmeans_v3.R', echo=TRUE)

Moment <- 12
accuracy.table <- accuracy.table.inv <- f1.table <- f1.table.inv <- matrix(NA, nrow = 6, ncol = (Moment+6))
time.user <- time.system <- time.elapsed <- matrix(NA, nrow = 6, ncol = 7)
colnames(time.user) <- colnames(time.system) <- colnames(time.elapsed) <- c("Data_Distribution", "SGT", "HMM", "MM", "SMM", "len_mean", "len_var")
colnames(f1.table) <- colnames(accuracy.table) <- colnames(f1.table.inv) <- colnames(accuracy.table.inv) <- c("Data_Distribution", "SGT1", "SGT2", "SGT3", "SGT4", "SGT5", "SGT6","SGT7", "SGT8", "SGT9", "SGT10", "SGT11", "SGT12", "HMM", "MM", "SMM", "len_mean", "len_var")




############# Random HMM data clustering #############
## Random HMM data generation
ii <- 1
accuracy.table.inv[ii,1] <- accuracy.table[ii,1] <- f1.table.inv[ii,1] <- f1.table[ii,1] <- time.user[ii,1] <- time.elapsed[ii,1] <- time.system[ii,1]<- "HMM"

library(seqHMM)
K = 5
n_states = 3
n_symbols = 6
ini_probs   <- simulate_initial_probs(n_states=n_states, n_clusters = K)
trans_probs <- simulate_transition_probs(n_states=n_states, n_clusters = K, left_right = FALSE, diag_c = 0)
emiss_probs <- simulate_emission_probs(n_states = n_states, n_symbols = n_symbols, n_clusters = K)

ss <- simulate_hmm(n_sequences = 10, initial_probs = ini_probs[[1]], transition_probs = trans_probs[[1]], emission_probs = emiss_probs[[1]], sequence_length = 25)

hmm.sequences <- NULL
for(k in 1:K)
{
  n_k <- f_rand_between(20,35,1)
  for(n in 1:n_k)
  {
    # l <- f_rand_between(10,45)  
    l <- f_rand_between(30,150)  
    ss <- simulate_hmm(n_sequences = 1, initial_probs = ini_probs[[k]], transition_probs = trans_probs[[k]], emission_probs = emiss_probs[[k]], sequence_length = l)
    seq <- toupper(letters[ss$observations$T1])
    seq <- f_merge(seq)
    hmm.sequences <- rbind(hmm.sequences, c(k, seq))    
  }
}

colnames(hmm.sequences) <- c("ActualCluster", "seq")
lens <- f_seq_len_mu_var(sequences = hmm.sequences)
accuracy.table.inv[ii,(Moment+5):(Moment+6)] <- accuracy.table[ii,(Moment+5):(Moment+6)] <- f1.table.inv[ii,(Moment+5):(Moment+6)] <- f1.table[ii,(Moment+5):(Moment+6)] <- time.user[ii,6:7] <- time.elapsed[ii,6:7] <- time.system[ii,6:7]<- c(lens$seq.lens.mu, lens$seq.lens.var)

ptm <- proc.time()
hmm.all_seq_mats<- f_all_seq_mats (sequence_all = hmm.sequences, lambda = n_symbols, decay = 1, skip_same_char = FALSE, M = Moment, trace = F)
tt <- proc.time() - ptm
time.user[ii, 2]    <- tt[1]/Moment
time.system[ii, 2]  <- tt[2]/Moment
time.elapsed[ii, 2] <- tt[3]/Moment

# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(hmm.sequences), all_seq_mats = hmm.all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize, "--HMM--")
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "HMM", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }

ptm <- proc.time()
for(m in 1:Moment)  ## SGT clustering without moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = hmm.all_seq_mats, length_normalize = TRUE, lambda = n_symbols, moment = m, inv.powered = F, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = n_symbols, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }
  
  cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table[ii, (m+1)]       <- cc$F1  
}
tt <- proc.time() - ptm
time.user[ii, 2]    <- as.numeric(time.user[ii, 2]) + tt[1]/Moment
time.system[ii, 2]  <- as.numeric(time.system[ii, 2]) + tt[2]/Moment
time.elapsed[ii, 2] <- as.numeric(time.elapsed[ii, 2]) + tt[3]/Moment

for(m in 1:Moment)  ## SGT clustering WITH moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = hmm.all_seq_mats, length_normalize = TRUE, lambda = n_symbols, moment = m, inv.powered = T, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = n_symbols, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }
  
  cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table.inv[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table.inv[ii, (m+1)]       <- cc$F1  
}

## HMM Clustering
# Step 1: Convert the sequences into the seqHMM compatible matrix
sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = hmm.sequences)

hidden_states <- n_states  # Arbitrary
n_symbols     <- n_symbols
K             <- K  # Number of clusters

ptm <- proc.time()
mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
tt <- proc.time() - ptm
time.user[ii, 3]    <- tt[1]
time.system[ii, 3]  <- tt[2]
time.elapsed[ii, 3] <- tt[3]

mhmm.out$mhmm.confusing_mat
accuracy.table[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1
accuracy.table.inv[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1

## Markov model clustering ##
ptm <- proc.time()
markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = hmm.sequences, K = K, lambda = n_symbols, algorithm = "mm", trace = F)
tt <- proc.time() - ptm
time.user[ii, 4]    <- tt[1]
time.system[ii, 4]  <- tt[2]
time.elapsed[ii, 4] <- tt[3]

markov_clusters.mm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
accuracy.table[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+3)]        <- cc$F1
accuracy.table.inv[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+3)]        <- cc$F1


ptm <- proc.time()
markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = hmm.sequences, K = K, lambda = n_symbols, algorithm = "smm", trace = F)
tt <- proc.time() - ptm
time.user[ii, 5]    <- tt[1]
time.system[ii, 5]  <- tt[2]
time.elapsed[ii, 5] <- tt[3]

markov_clusters.smm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K)
accuracy.table[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+4)]        <- cc$F1
accuracy.table.inv[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+4)]        <- cc$F1


########### Markov data part ##################
ii <- 2
accuracy.table.inv[ii,1] <- accuracy.table[ii,1] <- f1.table.inv[ii,1] <- f1.table[ii,1] <- time.user[ii,1] <- time.elapsed[ii,1] <- time.system[ii,1]<- "MM"

mm.M <- 6
K <- 5
sim.mm      <- f_sim_final(N = 100, K = K, M = mm.M+1, N_days = 10, H.type = "continuous", H.equal = T)  ## H.equal==T Makes it Markov with same holding time distribution ## +1 for the absorbing state.
sim.mm.data_raw <- data.frame(sim.mm$data_sim)
sim.mm.seq.data <- f_convert_smm2seq(data_raw = sim.mm.data_raw)
sim.mm.seq.data.single_state_rmvd <- f_rmv_single_state_sequences(sequence_all = sim.mm.seq.data)
sim.mm.sequences <- sim.mm.seq.data.single_state_rmvd
colnames(sim.mm.sequences) <- c('cluster','seq')
sim.mm.sequences <- sim.mm.sequences[order(sim.mm.sequences[,'cluster']),]

lens <- f_seq_len_mu_var(sequences = sim.mm.sequences)
accuracy.table.inv[ii,(Moment+5):(Moment+6)] <- accuracy.table[ii,(Moment+5):(Moment+6)] <- f1.table.inv[ii,(Moment+5):(Moment+6)] <- f1.table[ii,(Moment+5):(Moment+6)] <- time.user[ii,6:7] <- time.elapsed[ii,6:7] <- time.system[ii,6:7]<- c(lens$seq.lens.mu, lens$seq.lens.var)

## HMM Clustering
# Step 1: Convert the sequences into the seqHMM compatible matrix
sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = sim.mm.sequences)

hidden_states <- 2  # Arbitrary
n_symbols     <- mm.M
K             <- K  # Number of clusters

ptm <- proc.time()
mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
tt <- proc.time() - ptm
time.user[ii, 3]    <- tt[1]
time.system[ii, 3]  <- tt[2]
time.elapsed[ii, 3] <- tt[3]

mhmm.out$mhmm.confusing_mat
accuracy.table[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1
accuracy.table.inv[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1


## Markov clustering
ptm <- proc.time()
markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = sim.mm.sequences, K = K, lambda = mm.M, algorithm = "mm", trace = F)
tt <- proc.time() - ptm
time.user[ii, 4]    <- tt[1]
time.system[ii, 4]  <- tt[2]
time.elapsed[ii, 4] <- tt[3]

cc <- f_clustering_accuracy(actual = c(strtoi(sim.mm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
accuracy.table[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+3)]        <- cc$F1
accuracy.table.inv[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+3)]        <- cc$F1

ptm <- proc.time()
markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = sim.mm.sequences, K = K, lambda = mm.M, algorithm = "smm", trace = F)
tt <- proc.time() - ptm
time.user[ii, 5]    <- tt[1]
time.system[ii, 5]  <- tt[2]
time.elapsed[ii, 5] <- tt[3]

markov_clusters.smm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.mm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K)
accuracy.table[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+4)]        <- cc$F1
accuracy.table.inv[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+4)]        <- cc$F1

## SGT clustering
ptm <- proc.time()
mm.all_seq_mats<- f_all_seq_mats(sequence_all = sim.mm.sequences,long_seq = TRUE,long_seq_ele_limits = c(1,mm.M), lambda =mm.M, decay = 1, skip_same_char = FALSE, M = Moment, trace =F)
tt <- proc.time() - ptm
time.user[ii, 2]    <- tt[1]/Moment
time.system[ii, 2]  <- tt[2]/Moment
time.elapsed[ii, 2] <- tt[3]/Moment

ptm <- proc.time()
for(m in 1:Moment)  ## SGT clustering without moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = F, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }

  cc <- f_clustering_accuracy(actual = c(strtoi(sim.mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table[ii, (m+1)]       <- cc$F1  
}
tt <- proc.time() - ptm
time.user[ii, 2]    <- as.numeric(time.user[ii, 2]) + tt[1]/Moment
time.system[ii, 2]  <- as.numeric(time.system[ii, 2]) + tt[2]/Moment
time.elapsed[ii, 2] <- as.numeric(time.elapsed[ii, 2]) + tt[3]/Moment

for(m in 1:Moment)  ## SGT clustering WITH moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = T, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }

  cc <- f_clustering_accuracy(actual = c(strtoi(sim.mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table.inv[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table.inv[ii, (m+1)]       <- cc$F1  
}


# input_data <- f_create_input_kmeans(all_seq_mats = mm.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = 5, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = T)
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.mm.sequences[,1])), pred = c(class_k$class), K = K)
# cc



########### Semi-Markov data part ##################
ii <- 3
accuracy.table.inv[ii,1] <- accuracy.table[ii,1] <- f1.table.inv[ii,1] <- f1.table[ii,1] <- time.user[ii,1] <- time.elapsed[ii,1] <- time.system[ii,1]<- "SMM"

smm.M <- 6
K <- 4
sim.smm      <- f_sim_final(N = 1000, K = K, M = smm.M+1, N_days = 10, H.type = "discrete", H.equal = F)
sim.smm.data_raw <- data.frame(sim.smm$data_sim)
sim.smm.seq.data <- f_convert_smm2seq(data_raw = sim.smm.data_raw)
sim.smm.seq.data.single_state_rmvd <- f_rmv_single_state_sequences(sequence_all = sim.smm.seq.data)
sim.smm.sequences <- sim.smm.seq.data.single_state_rmvd
colnames(sim.smm.sequences) <- c('cluster','seq')
sim.smm.sequences <- sim.smm.sequences[order(sim.smm.sequences[,'cluster']),]

lens <- f_seq_len_mu_var(sequences = sim.smm.sequences)
accuracy.table.inv[ii,(Moment+5):(Moment+6)] <- accuracy.table[ii,(Moment+5):(Moment+6)] <- f1.table.inv[ii,(Moment+5):(Moment+6)] <- f1.table[ii,(Moment+5):(Moment+6)] <- time.user[ii,6:7] <- time.elapsed[ii,6:7] <- time.system[ii,6:7]<- c(lens$seq.lens.mu, lens$seq.lens.var)

## HMM Clustering
# Step 1: Convert the sequences into the seqHMM compatible matrix
sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = sim.smm.sequences)

hidden_states <- 3  # Arbitrary
n_symbols     <- smm.M
K             <- K  # Number of clusters

ptm <- proc.time()
mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
tt <- proc.time() - ptm
time.user[ii, 3]    <- tt[1]
time.system[ii, 3]  <- tt[2]
time.elapsed[ii, 3] <- tt[3]

mhmm.out$mhmm.confusing_mat
accuracy.table[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1
accuracy.table.inv[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1

# mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
# 
# mhmm.out$mhmm.confusing_mat

## Markov clustering
ptm <- proc.time()
markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = sim.smm.sequences, K = K, lambda = smm.M, algorithm = "mm", trace = F)
tt <- proc.time() - ptm
time.user[ii, 4]    <- tt[1]
time.system[ii, 4]  <- tt[2]
time.elapsed[ii, 4] <- tt[3]

markov_clusters.mm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
accuracy.table[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+3)]        <- cc$F1
accuracy.table.inv[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+3)]        <- cc$F1

ptm <- proc.time()
markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = sim.smm.sequences, K = K, lambda = smm.M, algorithm = "smm", trace = T)
markov_clusters.smm$likelihood_vec
markov_clusters.smm$clust_assign_diff_vec
time.user[ii, 5]    <- tt[1]
time.system[ii, 5]  <- tt[2]
time.elapsed[ii, 5] <- tt[3]

# write.csv(sim.smm.data_raw, 'simulated_smm_data.csv', row.names = F)

write.csv(cbind(markov_clusters.smm$likelihood_vec, markov_clusters.smm$clust_assign_diff_vec), 'tmp.csv', row.names = F)

markov_clusters.smm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K, do.permutation = T)

cc

reordered.classes <- f_reorder_class_assignment(class = markov_clusters.smm$clust$cluster)$consecutive_class
cc2 <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(reordered.classes), K = K, do.permutation = F)

accuracy.table[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+4)]        <- cc$F1
accuracy.table.inv[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+4)]        <- cc$F1

# SGT clustering
ptm <- proc.time()
smm.all_seq_mats<- f_all_seq_mats(sequence_all = sim.smm.sequences,long_seq = TRUE,long_seq_ele_limits = c(1,smm.M), lambda =smm.M, decay = 1, skip_same_char = FALSE, M = Moment, trace=F)
tt <- proc.time() - ptm
time.user[ii, 2]    <- tt[1]/Moment
time.system[ii, 2]  <- tt[2]/Moment
time.elapsed[ii, 2] <- tt[3]/Moment

ptm <- proc.time()
for(m in 1:Moment)  ## SGT clustering without moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = smm.all_seq_mats, length_normalize = TRUE, lambda = smm.M, moment = m, inv.powered = F, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = smm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }

  cc <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table[ii, (m+1)]       <- cc$F1  
}
tt <- proc.time() - ptm
time.user[ii, 2]    <- as.numeric(time.user[ii, 2]) + tt[1]/Moment
time.system[ii, 2]  <- as.numeric(time.system[ii, 2]) + tt[2]/Moment
time.elapsed[ii, 2] <- as.numeric(time.elapsed[ii, 2]) + tt[3]/Moment

for(m in 1:Moment)  ## SGT clustering WITH moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = smm.all_seq_mats, length_normalize = TRUE, lambda = smm.M, moment = m, inv.powered = T, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = smm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }
  cc <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table.inv[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table.inv[ii, (m+1)]       <- cc$F1  
}


# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(sim.smm.sequences), all_seq_mats = smm.all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize,"--SMM--")
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "Semi-Markov", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }
# 
# input_data <- f_create_input_kmeans(all_seq_mats = smm.all_seq_mats, length_normalize = TRUE, lambda = smm.M, moment = 10, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = 2, lambda = smm.M, trace = T)
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.smm.sequences[,1])), pred = c(class_k$class), K = K)
# cc


########### 2nd order Markov ##################
## First we develop the data
# Steps 1: Generate MM/SMM data with M^2+1 states, +1 for the absorbing
ii <- 4
accuracy.table.inv[ii,1] <- accuracy.table[ii,1] <- f1.table.inv[ii,1] <- f1.table[ii,1] <- time.user[ii,1] <- time.elapsed[ii,1] <- time.system[ii,1]<- "MM2"

mm.M <- 3
K <- 5
sim.2mm      <- f_sim_final(N = 100, K = K, M = mm.M^2+1, N_days = 10, H.type = "continuous", H.equal = T)  ## H.equal==T Makes it Markov with same holding time distribution

sim.2mm.data_raw <- data.frame(sim.2mm$data_sim)
sim.2mm.seq.data <- f_convert_smm2seq(data_raw = sim.2mm.data_raw)
sim.2mm.seq.data.single_state_rmvd <- f_rmv_single_state_sequences(sequence_all = sim.2mm.seq.data)
sim.2mm.sequences <- sim.2mm.seq.data.single_state_rmvd
colnames(sim.2mm.sequences) <- c('cluster','seq')
subs_map <- matrix(c(1,"A-A",
                     2,"A-B",
                     3,"A-C",
                     4,"B-A",
                     5,"B-B",
                     6,"B-C",
                     7,"C-A",
                     8,"C-B",
                     9,"C-C"), nrow=9, ncol =2, byrow=T)
sim.2mm.sequences <- f_substitute(sequence_all = sim.2mm.sequences, D = mm.M^2, subs_map = subs_map)
sim.2mm.sequences <- sim.2mm.sequences[order(sim.2mm.sequences[,'cluster']),]

lens <- f_seq_len_mu_var(sequences = sim.2mm.sequences)
accuracy.table.inv[ii,(Moment+5):(Moment+6)] <- accuracy.table[ii,(Moment+5):(Moment+6)] <- f1.table.inv[ii,(Moment+5):(Moment+6)] <- f1.table[ii,(Moment+5):(Moment+6)] <- time.user[ii,6:7] <- time.elapsed[ii,6:7] <- time.system[ii,6:7]<- c(lens$seq.lens.mu, lens$seq.lens.var)

## HMM Clustering
# Step 1: Convert the sequences into the seqHMM compatible matrix
sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = sim.2mm.sequences)

hidden_states <- 3  # Arbitrary
n_symbols     <- mm.M
K             <- K  # Number of clusters

ptm <- proc.time()
mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
tt <- proc.time() - ptm
time.user[ii, 3]    <- tt[1]
time.system[ii, 3]  <- tt[2]
time.elapsed[ii, 3] <- tt[3]

mhmm.out$mhmm.confusing_mat
accuracy.table[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1
accuracy.table.inv[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1


## Markov clustering
ptm <- proc.time()
markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = sim.2mm.sequences, K = K, lambda = mm.M, algorithm = "mm", trace = T)
tt <- proc.time() - ptm
time.user[ii, 4]    <- tt[1]
time.system[ii, 4]  <- tt[2]
time.elapsed[ii, 4] <- tt[3]

markov_clusters.mm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.2mm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
accuracy.table[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+3)]        <- cc$F1
accuracy.table.inv[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+3)]        <- cc$F1


ptm <- proc.time()
markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = sim.2mm.sequences, K = K, lambda = mm.M, algorithm = "smm", trace = T)
time.user[ii, 5]    <- tt[1]
time.system[ii, 5]  <- tt[2]
time.elapsed[ii, 5] <- tt[3]

markov_clusters.smm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.2mm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K)
accuracy.table[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+4)]        <- cc$F1
accuracy.table.inv[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+4)]        <- cc$F1

## SGT Clustering
ptm <- proc.time()
mm2.all_seq_mats<- f_all_seq_mats(sequence_all = sim.2mm.sequences, lambda =mm.M, decay = 1, lag = 1, skip_same_char = FALSE, M = Moment, trace = F)
tt <- proc.time() - ptm
time.user[ii, 2]    <- tt[1]/Moment
time.system[ii, 2]  <- tt[2]/Moment
time.elapsed[ii, 2] <- tt[3]/Moment

ptm <- proc.time()
for(m in 1:Moment)  ## SGT clustering without moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm2.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = F, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }
  
  cc <- f_clustering_accuracy(actual = c(strtoi(sim.2mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table[ii, (m+1)]       <- cc$F1  
}
tt <- proc.time() - ptm
time.user[ii, 2]    <- as.numeric(time.user[ii, 2]) + tt[1]/Moment
time.system[ii, 2]  <- as.numeric(time.system[ii, 2]) + tt[2]/Moment
time.elapsed[ii, 2] <- as.numeric(time.elapsed[ii, 2]) + tt[3]/Moment

for(m in 1:Moment)  ## SGT clustering WITH moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm2.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = T, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }
  
  cc <- f_clustering_accuracy(actual = c(strtoi(sim.2mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table.inv[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table.inv[ii, (m+1)]       <- cc$F1  
}

# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(sim.2mm.sequences), all_seq_mats = mm2.all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize,"--Markov 2nd order--")
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "Markov_2nd_order", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }
# 
# input_data <- f_create_input_kmeans(all_seq_mats = mm2.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = 10, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = 2, lambda = mm.M^2, trace = T)
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.2mm.sequences[,1])), pred = c(class_k$class), K = K)
# cc


########### 3rd order Markov ##################
## First we develop the data
ii <- 6
accuracy.table.inv[ii,1] <- accuracy.table[ii,1] <- f1.table.inv[ii,1] <- f1.table[ii,1] <- time.user[ii,1] <- time.elapsed[ii,1] <- time.system[ii,1]<- "MM3"

# Steps 1: Generate MM/SMM data with M^3+1 states, +1 for the absorbing
mm.M <- 3
K <- 5
sim.3mm      <- f_sim_final(N = 100, K = K, M = mm.M^3+1, N_days = 12, H.type = "continuous", H.equal = T)  ## H.equal==T Makes it Markov with same holding time distribution

sim.3mm.data_raw <- data.frame(sim.3mm$data_sim)
sim.3mm.seq.data <- f_convert_smm2seq(data_raw = sim.3mm.data_raw)
sim.3mm.seq.data.single_state_rmvd <- f_rmv_single_state_sequences(sequence_all = sim.3mm.seq.data)
sim.3mm.sequences <- sim.3mm.seq.data.single_state_rmvd
colnames(sim.3mm.sequences) <- c('cluster','seq')
subs_map <- matrix(c(1,"A-A-A",
                     2,"A-A-B",
                     3,"A-A-C",
                     4,"A-B-A",
                     5,"A-B-B",
                     6,"A-B-C",
                     7,"A-C-A",
                     8,"A-C-B",
                     9,"A-C-C",
                     10,"B-A-A",
                     11,"B-A-B",
                     12,"B-A-C",
                     13,"B-B-A",
                     14,"B-B-B",
                     15,"B-B-C",
                     16,"B-C-A",
                     17,"B-C-B",
                     18,"B-C-C",
                     19,"C-A-A",
                     20,"C-A-B",
                     21,"C-A-C",
                     22,"C-B-A",
                     23,"C-B-B",
                     24,"C-B-C",
                     25,"C-C-A",
                     26,"C-C-B",
                     27,"C-C-C"), nrow=27, ncol =2, byrow=T)
sim.3mm.sequences <- f_substitute(sequence_all = sim.3mm.sequences, D = mm.M^3, subs_map = subs_map)
sim.3mm.sequences <- sim.3mm.sequences[order(sim.3mm.sequences[,'cluster']),]

lens <- f_seq_len_mu_var(sequences = sim.3mm.sequences)
accuracy.table.inv[ii,(Moment+5):(Moment+6)] <- accuracy.table[ii,(Moment+5):(Moment+6)] <- f1.table.inv[ii,(Moment+5):(Moment+6)] <- f1.table[ii,(Moment+5):(Moment+6)] <- time.user[ii,6:7] <- time.elapsed[ii,6:7] <- time.system[ii,6:7]<- c(lens$seq.lens.mu, lens$seq.lens.var)

## HMM Clustering
# Step 1: Convert the sequences into the seqHMM compatible matrix
sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = sim.3mm.sequences)

hidden_states <- 2  # Arbitrary
n_symbols     <- mm.M
K             <- 5  # Number of clusters

ptm <- proc.time()
mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
tt <- proc.time() - ptm
time.user[ii, 3]    <- tt[1]
time.system[ii, 3]  <- tt[2]
time.elapsed[ii, 3] <- tt[3]

accuracy.table[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1
accuracy.table.inv[ii, (Moment+2)] <- mhmm.out$mhmm.confusing_mat$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+2)]        <- mhmm.out$mhmm.confusing_mat$F1

mhmm.out$mhmm.confusing_mat

## Markov clustering
ptm <- proc.time()
markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = sim.3mm.sequences, K = K, lambda = mm.M, algorithm = "mm", trace = T)
tt <- proc.time() - ptm
time.user[ii, 4]    <- tt[1]
time.system[ii, 4]  <- tt[2]
time.elapsed[ii, 4] <- tt[3]

markov_clusters.mm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
accuracy.table[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+3)]        <- cc$F1
accuracy.table.inv[ii, (Moment+3)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+3)]        <- cc$F1

# markov_clusters.mm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = K)
# cc

ptm <- proc.time()
markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = sim.3mm.sequences, K = K, lambda = mm.M, algorithm = "smm", trace = T)
time.user[ii, 5]    <- tt[1]
time.system[ii, 5]  <- tt[2]
time.elapsed[ii, 5] <- tt[3]

markov_clusters.smm$clust
cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K)
accuracy.table[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table[ii,(Moment+4)]        <- cc$F1
accuracy.table.inv[ii, (Moment+4)] <- cc$cc$overall["Accuracy"]
f1.table.inv[ii,(Moment+4)]        <- cc$F1

# markov_clusters.smm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = K)
# cc

## SGT Clustering
ptm <- proc.time()
mm3.all_seq_mats<- f_all_seq_mats(sequence_all = sim.3mm.sequences, lambda =mm.M, decay = 0.2, lag = 0, skip_same_char = FALSE, M = Moment, trace = F)
tt <- proc.time() - ptm
time.user[ii, 2]    <- tt[1]/Moment
time.system[ii, 2]  <- tt[2]/Moment
time.elapsed[ii, 2] <- tt[3]/Moment

ptm <- proc.time()
for(m in 1:Moment)  ## SGT clustering without moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm3.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = F, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }  

  cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table[ii, (m+1)]       <- cc$F1  
}
tt <- proc.time() - ptm
time.user[ii, 2]    <- as.numeric(time.user[ii, 2]) + tt[1]/Moment
time.system[ii, 2]  <- as.numeric(time.system[ii, 2]) + tt[2]/Moment
time.elapsed[ii, 2] <- as.numeric(time.elapsed[ii, 2]) + tt[3]/Moment


for(m in 1:Moment)  ## SGT clustering WITH moment inv powered
{
  input_data <- f_create_input_kmeans(all_seq_mats = mm3.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = m, inv.powered = T, trace = F)
  
  check <- 0
  while(check != K)
  {
    class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = F)
    check   <- length(levels(factor(class_k$class)))
  }  
  
  cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(class_k$class), K = K)
  
  accuracy.table.inv[ii, (m+1)] <- cc$cc$overall["Accuracy"]
  f1.table.inv[ii, (m+1)]       <- cc$F1  
}
save.image("beating7.RData")
write.csv(f1.table, "beating_f1.csv", row.names = F)
write.csv(accuracy.table, "beating_accuracy.csv", row.names = F)
write.csv(f1.table.inv, "beating_f1_inv.csv", row.names = F)
write.csv(accuracy.table.inv, "beating_accuracy_inv.csv", row.names = F)
write.csv(time.elapsed, 'time_elapsed.csv', row.names = F)
write.csv(time.system, 'time_system.csv', row.names = F)
write.csv(time.user, 'time_user.csv', row.names = F)

########### Plotting #############
## Run time
time.elapsed <- read.csv('Results/Beating/Master/time_elapsed.csv', header = T)
time.elapsed <- time.elapsed[1:5,1:5]

longData <- data.frame(melt(time.elapsed))
head(longData)

g1<-ggplot(data=longData, aes(x = Data_Distribution, y = value, fill=variable)) +  theme_bw() + 
  geom_bar(stat = "identity", width = 0.6, position=position_dodge(width = 0.7)) +
  scale_fill_discrete(name = "Fitted algorithm\n(distribution)")

# geom_bar(colour="black", stat = "identity", width = 0.6, position=position_dodge(width = 0.7)) +# scale_fill_grey(start = 0, end = 0.9, name = "Fitted algorithm\n(distribution)")
g1 <- g1 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                 panel.background = element_blank(), axis.line = element_line(colour = "black"),
                 axis.text=element_text(size=12),
                 axis.title=element_text(size=14,face="bold")) +
  xlab("Actual sequence distribution") +
  ylab("Runtime (in sec)")

print(g1)  

## F1-score
f1score <- read.csv('Results/Beating/Master/beating_f1_inv.csv', header = T)
f1score <- f1score[1:5,]
sgtf1 <- f1score[,2:13]


f1score.select <- data.frame(Data_Distribution = f1score[,1], SGT=do.call(pmax, sgtf1), HMM = f1score[,"HMM"], MM = f1score[,"MM"], SMM=f1score[,"SMM"])

longData <- data.frame(melt(f1score.select))
head(longData)

g2<-ggplot(data=longData, aes(x = Data_Distribution, y = value, fill=variable)) +  theme_bw() + 
  geom_bar(stat = "identity", width = 0.6, position=position_dodge(width = 0.7)) + 
  scale_fill_discrete(name = "Fitted algorithm\n(distribution)")
# geom_bar(colour="black", stat = "identity", width = 0.6, position=position_dodge(width = 0.7)) + 
# scale_fill_grey(start = 0, end = 0.9, name = "Fitted algorithm\n(distribution)")
g2 <- g2 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                 panel.background = element_blank(), axis.line = element_line(colour = "black"),
                 axis.text=element_text(size=12),
                 axis.title=element_text(size=14,face="bold")) +
  xlab("Actual sequence distribution") +
  ylab("F1-score")

print(g2)  
library(gridExtra)
grid.arrange(g2, g1, nrow = 1)


# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(sim.3mm.sequences), all_seq_mats = mm3.all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize,"--Markov 3rd order--")
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "Markov_3rd_order", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }
# 
# input_data <- f_create_input_kmeans(all_seq_mats = mm3.all_seq_mats, length_normalize = TRUE, lambda = mm.M, moment = 2, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = K, lambda = mm.M, trace = T)
# cc <- f_clustering_accuracy(actual = c(strtoi(sim.3mm.sequences[,1])), pred = c(class_k$class), K = K)
# cc
# 
# 
# ######### TRASH ##########
# library(caret)
# library(e1071)
# lvs <- c("normal", "abnormal")
# truth <- factor(rep(lvs, times = c(86, 258)),
#                 levels = rev(lvs))
# pred <- factor(
#   c(
#     rep(lvs, times = c(54, 32)),
#     rep(lvs, times = c(27, 231))),               
#   levels = rev(lvs))
# xtab <- table(pred, truth)
# confusionMatrix(xtab)
# confusionMatrix(pred, truth)
# x <- c(1,2,1)
# y <- c(1,1,2)
# cc <- confusionMatrix(x,y)
# names(cc)
# 
# 
# ## HMM clustering ##
# library(seqHMM)
# obs <- seqdef(data=hmm.sequences.disjoint[,2:101], start=2, labels=c("A","B","C"))
# init_mhmm <- build_mhmm(
#   observations = list(obs),
#   initial_probs = list(sc_init1, sc_init2),
#   transition_probs = list(sc_trans2, sc_trans1),
#   emission_probs = list(sc_emiss1, sc_emiss2),
#   cluster_names = c("Cluster 1", "Cluster 2"))
# 
# mhmm_fit <- fit_model(
#   init_mhmm, local_step = TRUE, threads = 1,
#   control_em = list(restart = list(times = 10)))
# mhmm <- mhmm_fit$model
# mhmm$most_probable_cluster
# summary(mhmm)
# pos <- posterior_probs(mhmm)
# mhmm.class <- matrix(NA, nrow=50,ncol=1)
# for(i in 1:50)
# {
#   if(sum(pos[1:2,,i])>sum(pos[3:4,,i])){
#     mhmm.class[i]<-1
#   }else{
#     mhmm.class[i]<-2
#   }
# }
# table(mhmm.class)
# 
# mhmmdata("biofam", package = "TraMineR")
# biofam_seq <- seqdef(
#   biofam[, 10:25], start = 15,
#   labels = c("parent", "left", "married", "left+marr", "child",
#              "left+child", "left+marr+ch", "divorced"))
# 
# f_get_f1(confusion = cc$cc$table)

############ Extra ##############
###### Arbitrary pattern simulation #######
#### The arbitrary patterns are like a non-parametric version of sequence data, in which similar sequences will have patches of common subsequences of any length and at any place
# 
# lambda             <- 26
# min.seq.len = 40; max.seq.len = 200; min.kmer.len=4; max.kmer.len = 12; max.kmers=9
# P = 2
# filler.length.upper <- 3
# decay <- 1
# Moments <- 12
# random.kmer.placement <- TRUE
# 
# 
# sim_out      <- f_data_sim(P=P, min.seq.len = min.seq.len, max.seq.len = max.seq.len, min.kmer.len = min.kmer.len, max.kmer.len = max.kmer.len, max.kmers = max.kmers, filler.length = c(1,filler.length.upper), lambda = lambda, random.kmer.placement = random.kmer.placement)
# 
# mean(sim_out$seq_fillers/sim_out$seq_len_all)
# f_seq_len_mu_var(sim_out$sequence_all)
# 
# ## SGT finding ##
# all_seq_mats <- f_all_seq_mats(sequence_all = sim_out$sequence_all, lambda = lambda, decay = decay, skip_same_char = FALSE, direction = 'forward', M = Moments)
# 
# # all_seq_mats_markov <- f_all_seq_mats_markov(sequence_all = sim_out$sequence_all, lambda = lambda, decay = decay, skip_same_char = skip_same_char)
# 
# normalize <- TRUE
# length_normalize <- TRUE      
# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(sim_out$sequence_all), all_seq_mats = all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize,"--Min kmer len=",min.kmer.len,"--Max kmer len=",max.kmer.len,"--Max kmers=",max.kmers,"--Min Seq len",min.seq.len, "--Max Seq len=",max.seq.len, "--Max filler len=",filler.length.upper)
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "arbitrary-pattern", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }
# 
# 
# 
# ## perform clustering and checking error
# ## SGT CLustering
# input_data <- f_create_input_kmeans(all_seq_mats = all_seq_mats, length_normalize = TRUE, lambda = lambda, moment = 2, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = 2, lambda = lambda, trace = T)
# cc <- f_clustering_accuracy(actual = c(strtoi(sim_out$sequence_all[,1])), pred = c(class_k$class), K = 2)
# cc
# 
# ## HMM Clustering
# # Step 1: Convert the sequences into the seqHMM compatible matrix
# sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = sim_out$sequence_all)
# 
# # Step 2: Perform clustering by assigning some parameter values
# hidden_states <- 4  # Arbitrary
# n_symbols     <- lambda
# K             <- 2  # Number of clusters
# 
# mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = lambda, K = K)
# 
# mhmm.out$mhmm.confusing_mat
# 
# ## MM/SMM clustering
# markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = sim_out$sequence_all, K = 2, lambda = lambda, algorithm = "mm", trace = T)
# markov_clusters.mm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(sim_out$sequence_all[,1])), pred = c(markov_clusters.mm$clust$cluster), K = 2)
# cc
# 
# markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = sim_out$sequence_all, K = 2, lambda = lambda, algorithm = "smm", trace = T)
# markov_clusters.smm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(sim_out$sequence_all[,1])), pred = c(markov_clusters.smm$clust$cluster), K = 2)
# cc

########### HMM sequence clustering ##############
### Simulating HMM sequences ###
# library(HMM)
# sc_init1 <- c(.3,.7)
# sc_trans1 <- matrix(c(.7,.3,
#                       .3,.7),2)
# sc_emiss1 <- matrix(c(.8,.1,.1,
#                       .2,.6,.1),2, byrow=T)
# hmm.model1 <- initHMM(c("x","y"), c("A","B","C"), sc_init1, sc_trans1, sc_emiss1)
# 
# simHMM(hmm.model1, length = 10)
# 
# sc_init2 <- c(.3,.7)
# sc_trans2 <- matrix(c(.8,.2,
#                       .2,.8),2)
# sc_emiss2 <- matrix(c(.1,.1,.8,
#                       .2,.1,.7),2, byrow=T)
# hmm.model2 <- initHMM(c("x","y"), c("A","B","C"), sc_init2, sc_trans2, sc_emiss2)
# hmm.sequences <- NULL
# hmm.sequences.disjoint <- matrix(NA, nrow = 50, ncol = 101)
# for(i in 1:25) #Cluster 1
# {
#   l <- f_rand_between(10,45)
#   hmm.seq <- simHMM(hmm.model1, length = l)
#   seq <- f_merge(hmm.seq$observation)
#   hmm.sequences <- rbind(hmm.sequences, c(1,seq))
#   hmm.sequences.disjoint[i,1:(l+1)] <- c(1, hmm.seq$observation)
# }
# for(i in 26:50) #Cluster 2
# {
#   l <- f_rand_between(10,45)
#   hmm.seq <- simHMM(hmm.model2, length = l)
#   seq <- f_merge(hmm.seq$observation)
#   hmm.sequences <- rbind(hmm.sequences, c(2,seq))
#   hmm.sequences.disjoint[i,1:(l+1)] <- c(2, hmm.seq$observation)
# }
# colnames(hmm.sequences) <- c('cluster','seq')
# 
# 
# hmm.all_seq_mats<- f_all_seq_mats (sequence_all = hmm.sequences, lambda = 3, decay = 1, skip_same_char = FALSE, direction = 'forward', M = Moments)
# 
# out          <- f_seq_all_mat_diff(sequence_all = NULL, N = nrow(hmm.sequences), all_seq_mats = hmm.all_seq_mats, Moments = Moments, normalize = normalize, length_normalize = length_normalize, terminal_element = NULL, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", moment_type = 'simple_moment')
# 
# 
# moment_all_diff  <- out$moment_all_diff 
# 
# for(moment in 1:Moments)
# {
#   chart_title   <- paste("Moment=", moment, "Clusters=",P, "--Decay=", decay,"--Normalize=",normalize, "--HMM--")
#   moment_plot   <- f_plot(diff_mat=moment_all_diff[,,moment], paste("Moment ", moment, "DIFF", chart_title))
#   mypath     <- file.path(getwd(),"results","beating", "HMM", paste("Moment_", moment, ".jpeg", sep = ""))
#   jpeg(file=mypath, width=1200, height=800)
#   print(moment_plot)
#   dev.off()
# }
# 
# input_data <- f_create_input_kmeans(all_seq_mats = hmm.all_seq_mats, length_normalize = TRUE, lambda = 3, moment = 10, normalize = TRUE, moment_type = 'simple_moment', trace = T)
# 
# class_k <- f_kmeans(input_data = input_data, K = 2, lambda = lambda, trace = T)
# class_k$class
# cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(class_k$class), K = 2)
# cc
# 
# ## HMM Clustering
# # Step 1: Convert the sequences into the seqHMM compatible matrix
# sequence_all.hmm <- f_get_seq2HMM_format(sequence_all = hmm.sequences)
# 
# hidden_states <- 2  # Arbitrary
# n_symbols     <- 3
# K             <- 2  # Number of clusters
# 
# mhmm.out <- f_mhmm_clustering(sequence_all.hmm = sequence_all.hmm, hidden_states = hidden_states, n_symbols = n_symbols, K = K)
# 
# mhmm.out$mhmm.confusing_mat
# 
# ## Markov model clustering ##
# markov_clusters.mm  <- f_sequence_markov_clustering(sequence_all = hmm.sequences, K = 2, lambda = lambda, algorithm = "mm", trace = T)
# markov_clusters.mm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(markov_clusters.mm$clust$cluster), K = 2)
# cc
# 
# markov_clusters.smm <- f_sequence_markov_clustering(sequence_all = hmm.sequences, K = 2, lambda = lambda, algorithm = "smm", trace = T)
# markov_clusters.smm$clust
# cc <- f_clustering_accuracy(actual = c(strtoi(hmm.sequences[,1])), pred = c(markov_clusters.smm$clust$cluster), K = 2)
# cc
# 
