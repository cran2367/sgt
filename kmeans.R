#### In this file, we have all the functions required for performing a k-means clustering. The k-means will be performed on the SGT-vectors. In there 'class' and 'cluster' of a sequence will mean the same.

f_centroid <- function(Ks, alphabet_set_size, input_data, class)
{
  # For any given classes, we find the centroids.
  # Inputs
  # Ks                  Vector of names of the classes. Typically, it is denoted by a scalar K, and cluster names are 1:K. But sometimes it's not 1:K (e.g. if one of the clusters are dropped mid way of clustering).
  # alphabet_set_size   The number of alphabets sequences are made of. 
  # input_data          A matrix, where a row is a SGT vector for a sequence and a column is one sgt feature.
  # class               A vector having class assignment for each sequence.
  
  K <- length(Ks)
  centroid <- matrix(rep(0, K * (alphabet_set_size * alphabet_set_size)), nrow = K)
  rownames(centroid) <- Ks
  
  for(k in Ks)
  {
    centroid[toString(k),] <- t(t(input_data) %*% (class == k) / sum(class == k))
  }
  
  return(centroid)
}


f_class <- function(Ks, n, input_data, centroid, asgnmt_threshold = 999999)
{
  # For given centroids we assign classes to each sequence
  # Inputs
  # Ks                  Vector of names of the classes. 
  # n                   Number of sequences
  # input_data          A matrix, where a row is a SGT vector for a sequence and a column is one sgt feature.
  # centroid            Matrix containing the centroids. Each row is a centroid for a cluster.
  # asgnmt_threshold    This assignment threshold is there for experimental purpose. Not needed for regular operations. Thus, it is default to a very high value.
  
  Z <- NULL
  for(k in Ks)
  {
    tmp <- input_data - t(matrix(rep(centroid[toString(k),], n), ncol = n))
    Z   <- cbind(Z, rowSums(abs(tmp)))  # We use L1 norm for distances.
  }
  
  colnames(Z) <- Ks
  class <- Ks[max.col(-1*Z, ties.method = "random")] # Updated classes by assigning the sequence to the class it is closest with
  
  wss  <- sum(abs(do.call(pmin,data.frame(Z))))
  
  class.last          <- max(as.numeric(Ks))
  threshold.violation <- c(do.call(pmin,data.frame(Z)) > asgnmt_threshold)
  if(any(threshold.violation))
  {
    class[threshold.violation] <- class.last + 1
    Ks <- c(Ks, (class.last + 1))
  }
  
  out <- list(class = class, Ks = Ks, wss = wss, Z = Z)
  return(out)
}


f_NA_centroid_exception <- function(Ks, centroid, trace = FALSE)
{
  # If there are too many classes, sometimes a class does not get any datapoint assigned, we should remove them. This is an important function to handle these exceptions.
  if(is.na(sum(sum(centroid))))
  {
    if(trace){print("inside nan")}
    centroid     <- centroid[!is.na(centroid[,1]), ] # Remove the centroid rows with Inf
    Ks           <- strtoi(rownames(centroid))
  }
  out <- list(Ks = Ks, centroid = centroid)
  
  return(out)
}

f_create_input_kmeans <- function(all_seq_sgt_parts, length_normalize = FALSE, alphabet_set_size, kappa, trace = TRUE, inv.powered = T)
{
  # Creating the input data for feeding into the kmeans function
  # Inputs
  # all_seq_sgt_parts       The transform on sequences in a dataset
  # length_normalize        Is True for length-insensitive variant of SGT [1]
  # alphabet_set_size       The number of alphabets that makes the sequences in the dataset.
  # kappa                   The tuning parameter
  # inv.powered             Is True if we want the take the kappa-th root of SGT as shown the algorithm 1 [1].
  
  n.seq  <- dim(all_seq_sgt_parts$W0_all)[3]
  
  # Find the SGT for each sequence
  sgt_mat_all <- array(rep(0,n.seq * alphabet_set_size * alphabet_set_size), 
                       dim=c(alphabet_set_size, alphabet_set_size, n.seq))
  
  for(ind in 1:n.seq)
  {
    if(trace){print(paste(ind,"in",n.seq))}
    if(length_normalize == TRUE)
    {
      sgt_mat_all[ , ,ind] <- f_SGT(W_kappa = all_seq_sgt_parts$W_kappa_all[[ind]], 
                                    W0 = all_seq_sgt_parts$W0_all[,,ind], 
                                    kappa = kappa, 
                                    Len = all_seq_sgt_parts$Len_all[ind],
                                    inv.powered = inv.powered)
    }else{ # Not length normalize
      sgt_mat_all[ , ,ind] <- f_SGT(W_kappa = all_seq_sgt_parts$W_kappa_all[[ind]], 
                                    W0 = all_seq_sgt_parts$W0_all[,,ind], 
                                    kappa = kappa, 
                                    Len = NULL,
                                    inv.powered = inv.powered)
    }
  }
  
  # Vectorize the sequence alphabet_set_size x alphabet_set_size statistics (mean in this case)
  # Code taken for this from http://stackoverflow.com/questions/4022195/transform-a-3d-array-into-a-matrix-in-r
  
  input_data      <- aperm(sgt_mat_all, c(3,2,1))
  dim(input_data) <- c(n.seq, alphabet_set_size * alphabet_set_size)
  
  return(input_data)
}


f_kmeans_procedure <- function(input_data, K, alphabet_set_size = 26, max_iteration = 50, trace = TRUE)
{
  # This function will perform the centroid based kmeans clustering using Manhattan distance.
  # Inputs
  # input_data      The input data matrix, each row a data point and the columns are its features
  # K               The number of clusters
  
  set.seed(12)  # To ensure reproducibility  
  n          <- nrow(input_data)
  
  # Step 0: Initialization
  if(K <= n) 
  {
    # Making sure at least one member is given to each cluster in the beginning
    class.tmp  <- 1:K
    class.tmp2 <- sample.int(n = K, size = (n - K), replace = T)
    class      <- c(class.tmp, class.tmp2)
    class.tmp2 <- sample.int(n = K, size = (n - K), replace = T)  # Another initialization for class.old
    class.old  <- c(class.tmp, class.tmp2)
  } else{
    stop("K is greater than n. Terminating!")
  }
  
  Ks       <- 1:K # List of cluster
  centroid <- f_centroid(Ks = Ks, alphabet_set_size = alphabet_set_size, input_data = input_data, class = class)
  
  out_NA   <- f_NA_centroid_exception(Ks = Ks, centroid = centroid, trace = trace)
  Ks       <- out_NA$Ks
  centroid <- out_NA$centroid
  
  # Iterations for clustering 
  class.changes <- 10 # arbitrary
  epsilon  <- 100 # arbitrary
  counter  <- 0
  class.changes.check <- 0
  
  while(class.changes != 0 && counter <= max_iteration)
  {
    counter   <- counter + 1
    class.old <- class
    
    # Step 1: Getting the centroid for each class
    centroid  <- f_centroid(Ks = Ks, alphabet_set_size = alphabet_set_size, input_data = input_data, class = class)
    
    # Exception handling: If a centroid does not get any data point assigned  
    out_NA    <- f_NA_centroid_exception(Ks = Ks, centroid = centroid)
    Ks        <- out_NA$Ks
    centroid  <- out_NA$centroid
    
    
    # Step 2: Assign (update) class to each data point based on its distance from the centroids
    class.out <- f_class(Ks = Ks, n = n, input_data = input_data, centroid = centroid)
    class <- class.out$class
    Ks    <- class.out$Ks
    wss   <- class.out$wss
    Z     <- class.out$Z
    
    # Iteration differences
    class.changes <- sum(class != class.old)
    
    if(trace)
    {
      print(paste("Iteration", counter, "in", max_iteration, "--Class chgs: ", class.changes, "wss: ", round(wss,2), "and K is ", length(Ks)))      
    }
  }
  return(list(class = class, centroid = centroid, Ks = Ks, wss = wss, Z = Z))
}


f_kmeans <- function(input_data, K, alphabet_set_size = 26, max_iteration = 50, trace = TRUE, K_fixed = T)
{
  if(K_fixed){
    
    check <- 0
    while(check != K)
    {
      class   <- f_kmeans_procedure(input_data = input_data, K = K, alphabet_set_size = alphabet_set_size, trace = trace)
      check   <- length(levels(factor(class$class)))
    }
  }else{
    class   <- f_kmeans_procedure(input_data = input_data, K = K, alphabet_set_size = alphabet_set_size, trace = trace)
  }
  return(class)
}


f_get_ss <- function(input_data)
{
  n  <- nrow(input_data)
  ybar <- t(rowMeans(t(input_data)))
  ss <- 0
  for(i in 1:n)
  {
    ss <- ss + sum(abs(input_data[i,] - ybar))
  }
  return(ss)
}


f_pcs <- function(input_data, PCs = 50)
{
  nc <- ncol(input_data)
  nr <- nrow(input_data)
  mu <- t(rowMeans(t(input_data)))
  Sigma <- cov(input_data)
  
  eg           <- eigen(Sigma)
  lam          <- eg$values
  lam.perc     <- lam/sum(lam)
  lam.perc.cum <- cumsum(lam.perc)
  
  print(paste(PCs, "PCs explain", round(lam.perc.cum[PCs]*100, 2),"percentage of variance"))
  
  V              <- eg$vectors
  tmp            <- sqrt(matrix(rep(lam, nc), nrow = nc, byrow = TRUE))
  V.norm         <- V / tmp
  V.norm.reduced <- V.norm[, 1:PCs]
  input_data_pcs <- (input_data - matrix(mu, nrow = nrow(input_data), ncol = nc)) %*% V.norm.reduced
  
  return(list(input_data_pcs = input_data_pcs, lam = lam, lam.perc.cum = lam.perc.cum))
}