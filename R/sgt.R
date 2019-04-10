## This lookup table will be needed throughout for converting integer to corresponding alphabet
alphabet_lookup <- data.frame(Integer=1:26, Alphabet = LETTERS)

####################################################################################
############### Algorithm 1: Parsing a sequence to obtain its SGT. #################
####################################################################################

f_sgt_parts <- function(sequence, kappa = 3, alphabet_set_size = 26, lag = 0, skip_same_char = FALSE, long_seq = FALSE, long_seq_ele_limits = NULL, spp = NULL)
{
  
  # The inputs
  # Sequence            Any given sequence (with padding)
  # kappa               The tuning param
  # alphabet_set_size   The number of alphabet_set the sequences are made up of
  # skip_same_char      Esp. for the element clustering it does not make sense to use repeated characters. For example, AAABC... in this actual transaction between A to B should be just one, however, if we don't skip the repetition it will be 3 (unnecessary inflating the transactions). Hence, skip repetition (value = TRUE) for element clustering.
  # long_seq            TRUE if sequences have many more alphabet_set (not just A-Z).
  # long_seq_ele_limits The limits of the alphabet_set in long sequences. For eg. in grayscale images it will be c(0,255)
    
  
  if(length(sequence) == 1)
  {
    s_split <- f_seq_split(sequence, spp = spp)
  } else{
    s_split <- sequence #already splitted
  }
  
  
  if(long_seq == FALSE)
  {
    rnames <- cnames <- c(levels(alphabet_lookup[,'Alphabet']))[1:alphabet_set_size]
  }else{
    alphabet_set_size <- long_seq_ele_limits[2] - long_seq_ele_limits[1] + 1
    rnames <- cnames <- seq(long_seq_ele_limits[1], long_seq_ele_limits[2], 1)
  }
  
  Len           <- length(s_split)
  
  Ls <- list()
  
  # Just the W0 corresponding to m = 0, and the M-th moment
  iter_set <- c(0, kappa)

  for(m in iter_set)  # The m=0 corresponds to W0
  {
    mat_Lm           <- matrix(rep(0,alphabet_set_size*alphabet_set_size), nrow=alphabet_set_size)
    rownames(mat_Lm) <- rnames
    colnames(mat_Lm) <- cnames
    
    for(i in 1:(Len-1))
    {
      if(skip_same_char == TRUE && s_split[i] == s_split[i+1])
      {
        # SKipping the loop when the next event in the sequence is same as current
        next
      }
      
      
      for(j in (i+1):length(s_split))
      {
        if(abs(j-i)>lag)
        {
          mat_Lm[s_split[i], s_split[j]]     <- mat_Lm[s_split[i], s_split[j]] + exp(-1*((abs(j-i) - lag)*m))
          
        }
      }
    }
    
    Ls[[length(Ls) + 1]] <- mat_Lm
  }
  
  output <- list(Len = Len, W0 = Ls[[1]], W_kappa = Ls[[2]])
  return(output)
}


##############################################################################################################
############### Algorithm 2: Extract SGT features by scanning alphabet positions of a sequence ###############
##############################################################################################################

f_get_alphabet_positions <- function(sequence_split, alphabet_set)
{
  # This function corresponds to the one defined in Line 1 in Algorithm 2 in [1]
  # Inputs
  # sequence_split   A sequence is passed as a vector of alphabets. It is called sequence split because a string sequence is split into its alphabets (in the same order)
  # alphabet_set    The set of alphabets sequence is made of.
  
  positions <- list()
  for(e in alphabet_set)
  {
    positions[[e]] <- which(sequence_split == e)
  }
  return(positions)
}

f_sgt_parts_using_alphabet_positions <- function(seq_alphabet_positions, alphabet_set, kappa = 12, lag = 0, skip_same_char = F)
{
  ### See the comments for the input parameters in f_seq_transform function.
  # seq_alphabet_positions      A list of index positions of all alphabet_set in the sequence
  # alphabet_set               Set of alphabet_set possible in the sequence. Can remove the long_seq and long_seq_ele_limits parameters because alphabet_set are given.
  
  Len <- sum(unlist(lapply(seq_alphabet_positions, function(x) length(x))))   # The sequence length
  
  Ls <- list()
  
  iter_set <- c(0, kappa)
  
  alphabet_set_size <- length(alphabet_set)
  for(m in iter_set)
  {
    mat_Lm           <- matrix(rep(0,alphabet_set_size*alphabet_set_size), nrow=alphabet_set_size)
    rownames(mat_Lm) <- colnames(mat_Lm) <- alphabet_set
    
    for(i in alphabet_set)
    {
      for(j in alphabet_set)
      {
        enumerated_combos <- arrange(expand.grid(i = seq_alphabet_positions[[i]],
                                                 j = seq_alphabet_positions[[j]]),
                                     i)
        x           <- c(enumerated_combos[,"j"]-enumerated_combos[,"i"])
        x.positives <- x[x>0]  # The x's which are greater than 0 are only corresponding to the feed-forward thing of the sequence. Others mean element j was before elemnet i.
        mat_Lm[i,j] <- sum(exp(-1*m*x.positives))  # Line 15 in Algorithm 2 in [1]
      }
    }
    
    Ls[[length(Ls) + 1]] <- mat_Lm
  }
  
  output <- list(Len = Len, W0 = Ls[[1]], W_kappa = Ls[[2]])
  return(output)
}


####################################################################################
##### Yield SGT output from the SGT parts computed from either algorithm 1 or 2 ####
####################################################################################

f_SGT <- function(W_kappa, W0, kappa, Len = NULL, inv.powered = T)
{
  ## This function computes the resulting SGT from the sgt parts found in function f_sgt_parts().
  # Inputs
  # W_kappa      See algorithm 1 in [1]
  # W0           See algorithm 1 in [1]
  # Len          Length of sequence
  # inv.powered  Is True if we want the take the kappa-th root of SGT as shown the algorithm 1 [1].
  
  if(!is.null(Len))# Normalizing for the length
  {
    W0 <- W0/Len
    W0[W0 == 0] <- NA
  }
  
  tmp <- W_kappa/W0  
  
  tmp[is.na(tmp)] <- 0
  
  SGT_mat <- tmp
  
  if(inv.powered){
    SGT_mat <- Math.invpow(SGT_mat, pow = kappa)  
  }
  
  return(SGT_mat)
}

  
f_SGT_for_each_sequence_in_dataset <- function(sequence_dataset, kappa = 3, alphabet_set = LETTERS, lag = 0, skip_same_char = FALSE, long_seq = FALSE, long_seq_ele_limits = NULL, spp = NULL, sgt_using_alphabet_positions = F, trace = T)
{
  # The inputs
  # Sequence_dataset    Either a vector with each element as a string (a sequence), or a dataframe with the sequences under column name 'seq'.
  # kappa               The tuning param
  # alphabet_set_size   The number of alphabet_set the sequences are made up of
  # skip_same_char      Esp. for the element clustering it does not make sense to use repeated characters. For example, AAABC... in this actual transaction between A to B should be just one, however, if we don't skip the repetition it will be 3 (unnecessary inflating the transactions). Hence, skip repetition (value = TRUE) for element clustering.
  # long_seq            TRUE if sequences have many more alphabet_set (not just A-Z).
  # long_seq_ele_limits The limits of the alphabet_set in long sequences. For eg. in grayscale images it will be c(0,255)
  # sgt_using_alphabet_positions
  #                     If True, then the alternate algorithm (Algorithm 2 in the paper) will be used.
  
  n.seq <- nrow(sequence_dataset)
  
  alphabet_set_size <- length(alphabet_set)
  
  Len_all <- array(rep(0,n.seq), dim = c(n.seq)) 
  W0_all  <- array(rep(0,n.seq*alphabet_set_size*alphabet_set_size), dim=c(alphabet_set_size,alphabet_set_size,n.seq))
  
  W_kappa_all <- list()
  
  if(ncol(sequence_dataset) > 1)
  {
    sequences <- sequence_dataset[, 'seq']  
  }else{
    sequences <- sequence_dataset
  }
  
  for(i in 1:n.seq)
  {
    if(trace){print(paste("Sequence",i,"of",n.seq))}
    
    if(!sgt_using_alphabet_positions)
    {
      sgt_parts <- f_sgt_parts(sequence = sequences[i], kappa = kappa, 
                          alphabet_set_size = alphabet_set_size, lag = lag, 
                          skip_same_char = skip_same_char, 
                          long_seq = long_seq, long_seq_ele_limits = long_seq_ele_limits)  
    }else{
      s_split                <- f_seq_split(sequence = sequences[i], spp = spp)
      seq_alphabet_positions <- f_get_alphabet_positions(sequence_split = s_split, alphabet_set = alphabet_set)
      sgt_parts              <- f_sgt_parts_using_alphabet_positions(seq_alphabet_positions = seq_alphabet_positions,
                                                                     alphabet_set = alphabet_set, 
                                                                     kappa = kappa,
                                                                     lag = lag, skip_same_char = skip_same_char)
    }
    
    tmp  <- sgt_parts$W0
    
    Len_all[i] <- sgt_parts$Len
    W0_all[, , i]   <- tmp
    
    W_kappa_all[[length(W_kappa_all) + 1]] <- sgt_parts$W_kappa
  }
  dimnames(W0_all) <- list(rownames(tmp), colnames(tmp), c(sequence_dataset[,1]))  
  
  output <- list(Len_all = Len_all, W0_all = W0_all, W_kappa_all = W_kappa_all)
  return(output)
}



################################################################################
############################   Auxiliary functions  ############################
################################################################################

## Get alphabet for an integer
f_get_alphabet <- function(integer)
{
  return(levels(factor(alphabet_lookup[alphabet_lookup[,'Integer']==integer, 'Alphabet'])))
}


f_seq_split <- function(sequence, spp = NULL)
{
  ## Split a sequence into a vector of alphabets. The order of alphabets is retained. Usually we get a sequence as a long string. This function just splits it to be further processed for SGT.
  ## Inputs
  # sequence   A sequence, e.g. "FSDFSFIFFSAOPDSA"
  # spp        The separator of alphabets in the sequence. In the above example it is NULL.
  
  ## Output
  # s_split    The input sequence returned as a vector of alphabets in the same order. 
  if(!is.null(spp))
  {
    tmp <- strsplit(x = sequence, split = spp)
    s_split <- tmp
  }else{
    countCharOccurrences <- function(char, s) {
      s2 <- gsub(char,"",s)
      return (nchar(s) - nchar(s2))
    }
    
    if(countCharOccurrences("-",sequence) > 1)
    {
      tmp <- strsplit(sequence, "-")
      s_split <- tmp
    }else if(countCharOccurrences("-",sequence) == 1)
    {
      tmp     <- strsplit(sequence, "-")
      tmp     <- tmp[[1]][1]
      s_split <- strsplit(tmp,"")
    }else if(length(grep(" ",sequence)))
    {
      s_split <- strsplit(sequence," ")
    }else if(length(grep("~",sequence)))
    {
      s_split <- strsplit(sequence,"~")
    }else
    {
      s_split <- strsplit(sequence,"")    
    }
  }
  
  s_split <- s_split[[1]]
  s_split <- s_split[s_split != ""]
  return(s_split)
}


Math.invpow <- function(x, pow) {
  sign(x) * abs(x)^(1/pow)
}

Math.pow <- function(x, pow) {
  out <- 1
  if(pow > 0){
    for(p in 1:pow)
    {
      out <- out * x
    }
  }else if(pow == 0){
    out <- 1
  }
  return(out)
}

Math.matrixpow <- function(x, pow) {
  out <- x
  if(pow > 0){
    for(p in 1:pow)
    {
      out <- out %*% x
    }
  }else if(pow == 0){
    out <- 1
  }
  return(out)
}

Math.matrix_norm <- function(mat, norm)
{
  if(norm == 1)
  {
    out <- abs(mat)
  }else{
    out <- mat^norm
  }
  return(out)
}

Math.standardize <- function(x, y) {
  y[y == 0] <- NA
  out <- x/y
  out[is.na(out)] <- 0
  return(out)
}


f_get_f1 <- function(confusion)
{
  ## In this function we find F1 score from a confusion matrix. This will be used to select a clustering model also in function f_clustering_accuracy()
  K <- ncol(confusion)
  f1 <- NULL
  for(k in 1:K)
  {
    tp <- confusion[k,k]  # True pos
    fp <- sum(confusion[, k])-confusion[k, k]  # False pos
    fn <- sum(confusion[k, ])-confusion[k, k]  # False neg
    tmp <- 2*tp / (2 * tp + fn + fp)
    f1 <- c(f1, tmp)
  }
  return (mean(f1))
}

f_clustering_accuracy <- function(actual, pred, K = 2, type = "f1", trace = F, do.permutation = T)
{
  ### In this function we will find the accuracy of clustering ffrom any clustering method.
  ## Inputs
  # actual   A vector of actual clusters
  # pred     A vector of estimated clusters
  # K        Number of clusters of classes
  # type     Best confusion selection method, type = c("accuracy", "f1")
  library(gtools)
  x <- letters[actual]
  y <- letters[pred]
  out_cc <- confusionMatrix(x,y)
  out_f1 <- NA
  if(type == "f1")
  {
    # out_f1 <- 2*(out_cc$byClass["Pos Pred Value"]*out_cc$byClass["Sensitivity"]/(out_cc$byClass["Pos Pred Value"]+out_cc$byClass["Sensitivity"]))  
    out_f1 <- f_get_f1(confusion = out_cc$table)
  }
  
  if(do.permutation)
  {
    possibilities <- permutations(K,K,letters[1:K]) # Depending on the version, one of these two lines (this and the one below) work
    # possibilities <- matrix(letters[permutations(K)], ncol = K)  
    
    ## We are trying for all possibilities because the digit of the assigned class does not matter. For that any naming is fine. Thus, end of the day, the one with the best accuracy is the right one.
    for(poss in 2:nrow(possibilities)){  # Number of other (hence, starting from 2) possibilities
      if(trace){print(paste("Trying possibility",poss,sep="-"))}
      for(k in 1:K){
        y[pred==k] <- possibilities[poss,k]
      }
      
      tmp <- confusionMatrix(x,y)
      if(type == "f1")
      {
        tmp_f1 <- f_get_f1(confusion = tmp$table)
        flag <- (tmp_f1 > out_f1)
      }else if(type == "accuracy"){
        flag <- (tmp$overall["Accuracy"] > out_cc$overall["Accuracy"])
      }
      
      if(flag){  # Choosing based on F1 score instead of accuracy
        out_cc <- tmp
        if(type=="f1"){
          out_f1 <- tmp_f1
          names(out_f1) <- "F1"      
        }
        
        if(trace){print(paste("Selecting possibility",poss,sep="-"))}
      }
    }  
  }
  
  return(list(cc = out_cc, F1=out_f1))  
}

f_reorder_class_assignment <- function(class)
{
  ## In this function we reorder the assigned classes in clustering, such that they are ordered with consecutive class labels for easier clustering accuracy check
  conse_class <- 1
  class_map <- matrix(c(class[1], conse_class), nrow = 1)
  final_class <- matrix(c(class[1], conse_class), nrow = 1)
  
  for(i in 2:length(class))
  {
    if(class[i] == class[i-1])
    {
      final_class <- rbind(final_class, cbind(class[i], class_map[class_map[,1]==class[i], 2]))
    }else{
      if(class[i] %in% class_map[,1])
      {
        final_class <- rbind(final_class, cbind(class[i], class_map[class_map[,1]==class[i], 2]))
      }else{
        conse_class <- conse_class + 1
        final_class <- rbind(final_class, cbind(class[i], conse_class))
        class_map   <- rbind(class_map, cbind(class[i], conse_class))
      }
    }
  }
  
  out <- list(class_mapped = final_class, consecutive_class = final_class[,2])
  return(out)
}


f_seq_len_mu_var <- function(sequences)
{
  # A function that will find the mean and var of the sequence lengths
  seq.lens <- NULL
  for(i in 1:nrow(sequences))
  {
    tmp <- f_seq_split(sequences[i,'seq'])
    seq.lens <- c(seq.lens, length(tmp))
  }
  seq.lens.mu  <- mean(seq.lens)
  seq.lens.var <- var(seq.lens)
  
  out <- list(seq.lens.mu = seq.lens.mu, seq.lens.var = seq.lens.var, seq.lens = seq.lens)
  return(out)
}
