## This lookup table will be needed throughout for converting integer to corresponding alphabet
## For installing Biostrings
# source("http://bioconductor.org/biocLite.R")
# biocLite("Biostrings")
source('libraries.R')
library("Biostrings")
alphabet_lookup <- read.csv('alphabet_lookup.csv',header=T)
# lambda          <- 26 # Only 21 types of characters

#### We want to simulate seqences. The question is how are we building them. We use known (but randomly generated) k-mers as building blocks for generating sequences for a cluster. For each cluster we will have a set of k-mers which we will randomly put one after another and intersperse some fillers in between. ####

# Inputs
# P                 Number of clusters
# max.kmer.length   Max length of kmers of a cluster
f_data_sim <- function(P = 5, min.seq.len = 40, max.seq.len = 100, min.kmer.len = 2, 
                       max.kmer.len = 8, min.kmers = 2, max.kmers=8, 
                       N.per.clust.min = 20, N.per.clust.max = 30,
                       filler.length = c(1,2), lambda = lambda, 
                       random.kmer.placement = FALSE, perform_kmer_overlapping = FALSE, 
                       num_kmer_overlaps = 0, long_seq = F, spp = "")
{
  kmer_all       <- NULL
  max.kmers      <- max.kmers
  N              <- rep(0,P)
  sequence_all   <- NULL
  class          <- NULL
  c              <- 0
  seq_len_all    <- NULL
  filler_len_all <- NULL
  seq_fillers    <- NULL
  
  # Running through each cluster
  for (p in 1:P)
  {
   # print(paste("Generating cluster--",p))
    # Make kmers
    if(p == 1 && perform_kmer_overlapping == TRUE)  # making sure the first cluster has max number of kmers that can be shared with other clusters
    {
      R <- max.kmers
    }else{
      R <- f_rand_between(min.kmers, max.kmers) # Anywhere between 2 to 8 kmer in a cluster  
    }
    
    kmer     <- f_get_kmers(R=R, min.kmer.len = min.kmer.len, max.kmer.len = max.kmer.len, kmer=kmer_null, lambda = lambda, long_seq = long_seq, spp = spp)
    
    if(perform_kmer_overlapping == TRUE && num_kmer_overlaps > 0)
    {
#       if(P != 2)
#       {
#         stop("P should be 2 when kmer overlapping")
#       }else{
        if(p >= 2) # note here we are matching small p, that is the current cluster
        {
          for(o in 1:num_kmer_overlaps)
          {
            if(o <= R)
            {
              kmer[o] <- kmer_all[(p - 1), o] # replace the new kmer with old kmers  
            }
          }
        }
#       }
    }

    kmer_all <- rbind(kmer_all,cbind(kmer, t(cbind(rep(NA,max.kmers - R)))))

    # Start generating sequences now
    N[p]           <- f_rand_between(N.per.clust.min, N.per.clust.max) # Number of sequences in cluster p
    c              <- c + 1 # increment the cluster number 
    class          <- c(class, rep(c, N[p])) # Store actual classes of sequences
    max.seq.length <- max.seq.len         # Max length of a sequence in any cluster
    sequence       <- NULL
   
    for (n in 1:N[p])
    {
      # Random length of the seq
      seq.length  <- f_rand_between(min.seq.len,max.seq.len)
      tmp         <- NULL; l <- 0;
      seq_len_all <- c(seq_len_all,seq.length)
      n.fillers   <- 0
      while (l < seq.length)
      {
        # Find the kmers present for this cluster
        if(length(which(is.na(kmer_all[p,])))!=0)
        {
          kmers.len <- min(which(is.na(kmer_all[p,])))-1
        }else{
          kmers.len <- length(kmer_all[p,])
        }
                          
        
        for(pk in 1:kmers.len)
        {
          # Start with a random filler and then keep adding fillers after putting each mer.
          filler.r       <- f_rand_between(filler.length[1],filler.length[2]) # Fillers between a mer(s)
          filler_len_all <- c(filler_len_all,filler.r)
          n.fillers      <- n.fillers + filler.r
          filler         <- NULL
          if(filler.r>0)
          {
            for (fr in 1:filler.r) { 
                if(!long_seq){filler <- paste(filler,f_get_alphabet(f_rand_between(1,lambda)),sep="")}else{
                  filler <- paste(filler,f_rand_between(1,lambda),sep=spp)
                }              
              
              } # Fillers with any letter between 1 and 21
            
            tmp      <- paste(tmp,filler,sep=spp)  
          }
          
          if(random.kmer.placement == FALSE) # If we want to place the kmers in order for each member of cluster
          {
            pick.mer <- kmer_all[p,pk]
          }else{ # If we want to place the kmers randomly 
            random.kmer.ind <- f_rand_between(1, kmers.len)
            pick.mer        <- kmer_all[p, random.kmer.ind]
          }
          
          tmp      <- paste(tmp,pick.mer,sep=spp)
          
          if(!long_seq){
            l        <- nchar(tmp)
          }else{
            l        <- length(f_seq_split(tmp))
          }
          
        }
      }
      
      seq_fillers <- c(seq_fillers, n.fillers)
      
      # seq.length.diff <- max.seq.length - nchar(tmp)
      # if(seq.length.diff>0)
      # {
      #   seq.padding <- NULL
      #   
      #   for(sl in 1:seq.length.diff) {seq.padding <- paste(seq.padding,"*",sep="")}
      #   tmp <- paste(tmp, seq.padding, sep="-")
      # } else
      # {
      #   tmp <- substr(tmp, 1, max.seq.length)
      # }
      # 
      if(long_seq){tmp <- gsub("--","-", as.character(tmp))}
      sequence <- rbind(sequence,tmp)
    } # end of n loop (sequence in a cluster)
    rownames(sequence) <- NULL
    
    sequence_all <- rbind(sequence_all, cbind(rep(p, nrow(sequence)), sequence))
  } # end of p loop (cluster)
  colnames(kmer_all) <- NULL
  colnames(sequence_all) <- c("ActualCluster","seq")
  return(list(sequence_all = sequence_all, seq_fillers = seq_fillers, filler_len_all = filler_len_all, seq_len_all = seq_len_all, actual_class = class, kmer_all = cbind(1:P,kmer_all)))
}

## Generate k-mers. Input is R, the number of k-mers to make. Also supply a null kmer
f_get_kmers <- function(R, min.kmer.len, max.kmer.len, kmer, lambda = lambda, long_seq = F, spp = "")
{
  kmer <- NULL
  
  for (r in 1:R)
  {
    merlength <- f_rand_between(min.kmer.len, max.kmer.len) # Length of kmer can 
    
    tmp <- NULL
    
    if(merlength > 0)
    {
      for (me in 1:merlength)
      {
        i <- f_rand_between(1,lambda)
        
        if(!long_seq){
          s <- f_get_alphabet(i)
          tmp  <- paste(tmp, s, sep=spp)
        }else{
          tmp  <- paste(tmp, i, sep = spp)
        }
      }  
    }else if(merlength == 0)
    {
      tmp <- ""
    }

    kmer <- cbind(kmer, tmp)
  }
  return(kmer)
}

## Get alphabet for an integer
f_get_alphabet <- function(integer)
{
  return(levels(factor(alphabet_lookup[alphabet_lookup[,'Integer']==integer, 'Alphabet'])))
}

## Generate unif integer random number between (a,b)
f_rand_between <- function(a,b, n = 1)
{
  # function to find integer random number between a and b. n is the number of generated integers
  # return(floor(a + (b-a) * runif(n)))
  return(sample(a:b, size = n, replace = TRUE))
}

## Now for a given sequence we should compute the summary statistic matrices
# For a given sequence we have to find the matrices by scanning through the sequence and convert it into summary statistic matrices. We have stored the sequences as equal lengths by padding shorter matrices with padding of "X" separated by "-". Hence below we take care of extracting the actual sequence first.
# We start with first spliyt a string of sequence into a vector of characters
f_seq_split <- function(sequence, spp = NULL)
{
#   if(strsplit(sequence, "")[[1]][1] == "-")
#   {
#     tmp <- strsplit(sequence, "-")
#     tmp <- tmp[[1]]
#     s_split <- tmp[2:length(tmp)]
#   }else
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

f_seq_transform <- function(sequence, lambda = 26, decay=0.5, lag = 0, skip_same_char = FALSE, M = 3, long_seq = FALSE, long_seq_ele_limits = NULL, just.one.moment = F, spp = NULL)
{
  # Note: In this cleaned version, we remove the trial things, like, direction, getLs, log_inverse..., to clean and fasten the process. Also, the case, decay == False, is removed
  # The inputs
  # Sequence        Any given sequence (with padding)
  # lambda          The number of elements the sequences are made up of
  # decay           We take distance as decaying with exponential function exp(-lambda*(j-i))
  # skip_same_char  Esp. for the element clustering it does not make sense to use repeated characters. For example, AAABC... in this actual transaction between A to B should be just one, however, if we don't skip the repetition it will be 3 (unnecessary inflating the transactions). Hence, skip repetition (value = TRUE) for element clustering.
  # M               The number of moments we are computing
  # direction       options c('forward', 'backward', 'both'). Where 'forward' implies we look in the forward direction while finding the elements next to one another. Similarly for 'backward' and in case of 'both' we look in both direction. Default is 'forward' and is tested to work in all tried conditions.
  # long_seq        TRUE if sequences have many more elements (not just A-Z)
  # long_seq_ele_limits
  #                 The limits of the elements in long sequences. For eg. in grayscale images it will be c(0,255)   
  # getLs           It's a quick fix for this code. For large image sequences having redundant looping is too much waste of time
  # log_inverse_transform_for_xtra_long       This will tell if we have to apply this transform on the extracted features or not - mat_sum[mat_sum > 0] <- 1/(log(mat_sum[mat_sum > 0])). We saw its need while dealin with image_analysis when the extracted features were just coming as 0.
  # just.one.moment  If it is TRUE, it means we just have to get the M-th moment, and not 1 to M moments
  
  if(length(sequence) == 1)
  {
      s_split <- f_seq_split(sequence, spp = spp)
  } else{
    s_split <- sequence #already splitted
  }
  

  if(long_seq == FALSE)
  {
    rnames <- cnames <- c(levels(alphabet_lookup[,'Alphabet']))[1:lambda]
  }else{
    lambda <- long_seq_ele_limits[2] - long_seq_ele_limits[1] + 1
    rnames <- cnames <- seq(long_seq_ele_limits[1], long_seq_ele_limits[2], 1)
  }

  Len           <- length(s_split)
  alphabets_vec <- Len   # We do not really need alphabet vector, just the sequence length

  Ls <- list()
  
  if(!just.one.moment)  # Get all Ls (Moments)
  {
    M_set <- 0:M
  }else if(just.one.moment){  # Just the mat_count corresponding to m = 0, and the M-th moment
    M_set <- c(0, M)
  }
  
  for(m in M_set)  # The m=0 corresponds to mat_count
  {
    mat_Lm           <- matrix(rep(0,lambda*lambda), nrow=lambda)
    rownames(mat_Lm) <- rnames
    colnames(mat_Lm) <- cnames
    
    for(i in 1:(Len-1))
    {
      # if(Len > 100){print(paste(i,"in",Len))}
      if(skip_same_char == TRUE && s_split[i] == s_split[i+1])
      {
        # SKipping the loop when the next event in the sequence is same as current
        # print(paste('Repetition of-', s_split[i]))
        next
      }
      
      
      for(j in (i+1):length(s_split))
      {
        if(abs(j-i)>lag)
          {
          mat_Lm[s_split[i], s_split[j]]     <- mat_Lm[s_split[i], s_split[j]] + exp(-1*decay*((abs(j-i) - lag)*m))
          
          }
       }
     }

      Ls[[length(Ls) + 1]] <- mat_Lm
  }
  
  output <- list(alphabets_vec = alphabets_vec, mat_count = Ls[[1]], Ls = Ls[2:length(Ls)])
  return(output)
}

## Now, we want to parse all sequences and scan them to get their matrix forms.
f_all_seq_mats <- function(sequence_all, lambda = 26, decay = 0.5, lag = 0, skip_same_char = FALSE, M = 3, long_seq = FALSE, long_seq_ele_limits = NULL, trace = TRUE, just.one.moment = F, use_f_seq_transform_using_element_positions = F, elements = NULL, spp = NULL)
{
  n.seq <- nrow(sequence_all)

  alphabets_vec_all <- array(rep(0,n.seq), dim = c(n.seq)) 
  mat_count_all     <- array(rep(0,n.seq*lambda*lambda), dim=c(lambda,lambda,n.seq))
  
  Ls_all <- list()
  
  if(ncol(sequence_all) > 1)
  {
    sequences <- sequence_all[, 'seq']  
  }else{
    sequences <- sequence_all
  }
  sequences <- sequence_all[, 'seq']  
  for(i in 1:n.seq)
  {
    if(trace){print(paste("Sequence",i,"of",n.seq))}
    
    if(!use_f_seq_transform_using_element_positions)
    {
      mats <- f_seq_transform(sequence = sequences[i], lambda = lambda, decay = decay, lag = lag, skip_same_char = skip_same_char, M = M, long_seq = long_seq, long_seq_ele_limits = long_seq_ele_limits, just.one.moment = just.one.moment)  
    }else{
      s_split <- f_seq_split(sequence = sequences[i,'seq'], spp = spp)
      sequence_ele_positions <- f_get_element_positions(sequence_split = s_split, elements = elements)
      mats <- f_seq_transform_using_element_positions(sequence_ele_positions = sequence_ele_positions, elements = elements, M = M, decay = decay, lag = lag, skip_same_char = skip_same_char, just.one.moment = just.one.moment)
    }
    
    tmp  <- mats$mat_count
    
    alphabets_vec_all[i] <- mats$alphabets_vec
    mat_count_all[, , i]   <- tmp
    
    Ls_all[[length(Ls_all) + 1]] <- mats$Ls
  }
  dimnames(mat_count_all) <- list(rownames(tmp), colnames(tmp), c(sequence_all[,1]))  
  # dimnames(alphabets_vec_all) <- list(rownames(tmp), c(sequence_all[,1]))  
  
  output <- list(alphabets_vec_all = alphabets_vec_all, mat_count_all = mat_count_all, Ls_all = Ls_all)
  return(output)
}

f_mat_moment <- function(Ls, zzz = NULL, mat_count, n, alphabets_vec = NULL, moment_type = 'simple_moment', just.one.moment = F)
{
  # We dropped the unnecessary arguments for terminal element
  # Input
  # n     The moment you're interested in
  # type   markov: first order markov transition probability. Simply rowsum = 1  -- REMOVED
  #        simple_count: Simply count averaged by length  -- REMOVED
  #        simple_moment: Simple n-th moment, no integrations of 1-n transforms
  # just.one.moment   Here we are taking just n-th moment anyway, however, if just.one.moment was TRUE while getting the sequence transform, Ls will have just length 1. So we should draw the index 1 from Ls if it is TRUE.
  
  
  if(!is.null(alphabets_vec))# Normalizing for the length
  {
    mat_count <- mat_count/alphabets_vec
    mat_count[mat_count == 0] <- NA
  }
  
  if(!just.one.moment)
  {
    tmp <- Ls[[n]]/mat_count  
  }else if(just.one.moment){
    tmp <- Ls[[1]]/mat_count  
  }
  
    
  tmp[is.na(tmp)] <- 0
  mat_moment <- tmp
  mat_moment[is.na(mat_moment)] <- 0
    
  return(mat_moment)
}


f_all_seq_moments <- function(sequence_all = NULL, all_seq_mats = NULL, lambda = 26, decay = 1, Moments = 10, length_normalize = FALSE, skip_same_char = TRUE, long_seq = FALSE, long_seq_ele_limits = NULL, get_aggregate = FALSE)
{
  if(is.null(sequence_all) && is.null(all_seq_mats))
  {
    stop('Error: Insufficient Inputs')
  }else if(is.null(all_seq_mats))
  {
    all_seq_mats <- f_all_seq_mats(sequence_all, lambda = lambda, decay = decay, skip_same_char = skip_same_char, M = Moments, long_seq = long_seq, long_seq_ele_limits = long_seq_ele_limits)
    n.seq <- nrow(sequence_all)
  }else{
    n.seq <- dim(all_seq_mats$mat_count_all)[3]
  }
  
  if(long_seq == FALSE)
  {
    rnames <- cnames <- c(levels(alphabet_lookup[,'Alphabet']))[1:lambda]
  }else{
    lambda <- long_seq_ele_limits[2] - long_seq_ele_limits[1] + 1
    rnames <- cnames <- seq(long_seq_ele_limits[1], long_seq_ele_limits[2], 1)
  }
  print(n.seq)
  mat_moments_all <- array(rep(0,n.seq*lambda*lambda), dim=c(lambda,lambda,n.seq))
  if(get_aggregate)
  {
    mat_moments_agg <- matrix(0, ncol=lambda, nrow = lambda)
  }else{
    mat_moments_agg <- NULL
  }
  
  for (i in 1:n.seq)
  {
    print(paste("Momenting",i,"in",n.seq))
    if(length_normalize == TRUE)
    {
      mat_moments_all[ , ,i] <- f_mat_moment(Ls = all_seq_mats$Ls_all[[i]], mat_count = all_seq_mats$mat_count_all[,,i], n = Moments, alphabets_vec = all_seq_mats$alphabets_vec_all[i])  
    }else{
      mat_moments_all[ , ,i] <- f_mat_moment(Ls = all_seq_mats$Ls_all[[i]], mat_count = all_seq_mats$mat_count_all[,,i], n = Moments)  
    }
    
    if(get_aggregate)
    {
      mat_moments_agg <- mat_moments_agg + matrix(Math.invpow(mat_moments_all[,,i], pow = Moments), ncol = lambda, nrow = lambda)
    }
  }
  return(list(mat_moments_all = Math.invpow(mat_moments_all, pow = Moments), mat_moments_agg = mat_moments_agg))
}



f_seq_diff_btw_moment <- function(ind1, ind2, sequence_all, all_seq_mats, moment, length_normalize = FALSE, inv.powered = T)
{
  # Advanced version of f_seq_diff_btw for a specific moment
  alphabets_vec_all <- all_seq_mats$alphabets_vec_all
  mat_count_all     <- all_seq_mats$mat_count_all
  Ls_all            <- all_seq_mats$Ls_all
  
  if(length_normalize == TRUE)
  {
    seq1.moment <- f_mat_moment(Ls = Ls_all[[ind1]], mat_count = mat_count_all[,,ind1],n = moment, alphabets_vec = alphabets_vec_all[ind1])
  }else{
    seq1.moment <- f_mat_moment(Ls = Ls_all[[ind1]], mat_count = mat_count_all[,,ind1],n = moment)
  }

  alpha1 <- alphabets_vec_all[ind1]

  if(length_normalize == TRUE)
  {
    seq2.moment <- f_mat_moment(Ls = Ls_all[[ind2]], mat_count = mat_count_all[,,ind2],n = moment, alphabets_vec = alphabets_vec_all[ind2])
  }else{
    seq2.moment <- f_mat_moment(Ls = Ls_all[[ind2]], mat_count = mat_count_all[,,ind2],n = moment)
  }

  alpha2 <- alphabets_vec_all[ind2]
  
  if(inv.powered)
  {
    pow = moment
  }else{
    pow = 1
  }
  # diff <- (Math.invpow(seq1.moment, pow = 1) - Math.invpow(seq2.moment, pow = 1))
    diff <- (Math.invpow(seq1.moment, pow = pow) - Math.invpow(seq2.moment, pow = pow))
  
  total_diff <- (sum(sum(abs(diff))))

  return(total_diff)
}

f_seq_identity <- function(s1, s2, type = "global", gapOpening = 0, gapExtension = 1)
{
  # finds identity between two sequences
#   library("Biostrings")
  s1 <- f_merge(f_seq_split(s1))
  s2 <- f_merge(f_seq_split(s2))
  pwa      <- pairwiseAlignment(pattern = s1, subject = s2, type = type, gapOpening = gapOpening, gapExtension = gapExtension)
  identity <- nmatch(pwa) / nchar(pwa)
  return(identity)
}

## Generate a n x n (n = # sequences) matrix with mean and var matrix differences among each other
f_seq_all_mat_diff <- function(sequence_all = NULL, N = NULL, all_seq_mats, Moments, length_normalize = FALSE, get_skg_diff = TRUE, get_identity = FALSE, alignment_type = "global", trace = TRUE, inv.powered = T)
{
  if(is.null(sequence_all))
  {
    n.seq <- N
  }else{
    n.seq   <- nrow(sequence_all)
  }
  #n.seq   <- nrow(sequence_all)
  #rnames  <- sequence_all[,1] # Uncomment for protein real data
  rnames  <- 1:n.seq # Uncomment for simulation data

  seq_all_identity <- matrix(0, nrow = n.seq, ncol = n.seq, dimnames = list(rnames, rnames))
  
  moment_all_diff <- array(rep(0, n.seq*n.seq*Moments), dim = c(n.seq,n.seq,Moments), dimnames = list(rnames, rnames, c(1:Moments)))
  
  for(i in 1:(n.seq - 1))
  {
    if(trace){print(paste(i,"in",n.seq))}
    #for(j in 1:n.seq)
    for(j in (i+1):n.seq)
    {
      if(get_skg_diff == TRUE)
      {
        for(moment in 1:Moments)
        {
          moment_all_diff[i,j,moment] <- f_seq_diff_btw_moment(ind1 = i, ind2 = j, sequence_all = sequence_all, all_seq_mats = all_seq_mats, moment = moment, length_normalize = length_normalize, inv.powered = inv.powered)
        }  
      }
      
      if(get_identity == TRUE)
      {
        seq_all_identity[i,j] <- f_seq_identity(s1 = sequence_all[i,"seq"], s2 = sequence_all[j,"seq"], type = alignment_type)  
      }
    }
  }
  

  if(get_skg_diff == TRUE)
  {
    for(moment in 1:Moments)
    {
      moment_all_diff[,,moment] <- moment_all_diff[,,moment] + t(moment_all_diff[,,moment])
    }  
  }
  
  if(get_identity == TRUE)
  {
    seq_all_identity <- seq_all_identity + t(seq_all_identity)  
  }
  
  if(get_identity == TRUE)
  {
    out <- list(moment_all_diff = moment_all_diff, seq_all_identity = seq_all_identity)  
  }else{
    out <- list(moment_all_diff = moment_all_diff)
  }
  
  return(out)
}

## In this function we take in a difference matrix, can be mean diff or var diff matrix. We melt it into a longdata as needed by ggplot and then plot
f_plot <- function(diff_mat, chart_title = 'Test', af.size = 12)
{
  longData <- data.frame(melt(diff_mat))
  head(longData)
  
  myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")), space="Lab")
  # Simple ggplot2 heatmap
  # with colorBrewer "spectral" palette
  colnames(longData) <- c("x1","x2","value")
  
  # Probably heat map can't handle negative values, hence shifting the values.
  if(min(longData[,'value']) < 0){
    longData[,'value'] <- longData[,'value'] + abs(min(longData[,'value']))
  }

  zp1 <- ggplot(longData,
                aes(x = x1, y = x2, fill = value))
  zp1 <- zp1 + geom_tile(colour="white")
  # zp1 <- zp1 + scale_fill_gradientn(colours = myPalette(100))
  zp1 <- zp1 + scale_fill_gradientn(colours = myPalette(100),
                                    name = "Edge Weight")
  # zp1 <- zp1 + scale_fill_gradient(low = "white", high = "steelblue")
  zp1 <- zp1 + scale_x_discrete(expand = c(0, 0))
  zp1 <- zp1 + scale_y_discrete(expand = c(0, 0))
  zp1 <- zp1 + coord_equal()
  zp1 <- zp1 + theme_bw() + ggtitle(chart_title) + theme(axis.text.x = element_text(angle = 45, hjust = 1, size=af.size), 
                                                         axis.text.y = element_text(size = af.size),
                                                         axis.title.x=element_blank(),
                                                         axis.title.y=element_blank()
                                                        )
  #print(zp1) 
  
  return(zp1)  
}

## In this function we take in a difference matrix, and plot a jumbled plot to show how noisy the data is
f_plot_dump <- function(diff_mat, chart_title)
{
  longData <- data.frame(melt(diff_mat))
  head(longData)
  index <- 1:nrow(longData)
    
  index <- sample(index)
  longData$index <- index
  longData <- longData[order(longData[,'index']),]
  colnames(longData) <- c("x1","x2","value","index")
  longData$x1 <- as.character(longData$x1)
  longData$x2 <- as.character(longData$x2)
  
  myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")), space="Lab")
  # Simple ggplot2 heatmap
  # with colorBrewer "spectral" palette
 
  
  zp1 <- ggplot(longData,
                aes(x = x1, y = x2, fill = value))
  zp1 <- zp1 + geom_tile(colour="white")
  zp1 <- zp1 + scale_fill_gradientn(colours = myPalette(100))
  # zp1 <- zp1 + scale_fill_gradient(low = "white", high = "steelblue")
  zp1 <- zp1 + scale_x_discrete(expand = c(0, 0))
  zp1 <- zp1 + scale_y_discrete(expand = c(0, 0))
  zp1 <- zp1 + coord_equal()
  zp1 <- zp1 + theme_bw() + ggtitle(chart_title)
  # print(zp1) 
  
  return(zp1)  
}

## Now we want to compare our results with a crude method of finding sequence similarities by matching sequence alignments. We simply compare whether the elements match one by one.
f_crd_seq_diff <- function(seq1, seq2) # Between two seq
{
  s1 <- f_seq_split(seq1)
  s2 <- f_seq_split(seq2)
  
  diff <- abs(length(s1) - length(s2)) + sum(s2!=s1)
  
  return(diff)
}

f_crd_seq_diff_all <- function(sequence_all) # Between all using above f_crd_seq_diff function
{
  n.seq <- nrow(sequence_all)
  crd_diff <- matrix(rep(0,n.seq*n.seq), ncol=n.seq)
  for(i in 1:n.seq)
  {
    for(j in 1:n.seq)
    {
      crd_diff[i,j] <- f_crd_seq_diff(sequence_all[i,2],sequence_all[j,2])
    }
  }
  return(crd_diff)
}

## In this function we take in one of the distance matrix between all sequences and perform hierarchical clustering. We also need to provide the number of clusters as an input
f_clustering <- function(diff_mat, nclust, method="average")
{
  dist_mat <- as.dist(diff_mat)
  tree     <- hclust(dist_mat, method=method)
  class    <- cutree(tree, k = nclust)
  out      <- cbind((colnames(diff_mat)), class)
  colnames(out) <- c("sequence", "class")
  return(out)
}

## This function will find the optimal number of clusters for a given data. We will extend this function later to also find the optimal moment to use for clustering. The input 'data' here is matrix with each row for a data point and each column for an attribute.
f_optimal_clusters <- function(data, max_clust = 10, method = "kmeans")
{
  out <- NULL
  for(k in 2:max_clust)
  {
    (cl <- Kmeans(data, k, method = "manhattan"))
    out <- rbind(out, c(k,sum(cl$withinss/(2*cl$size))))
  }
  
  p <- plot(out[,1], out[,2], type = "b", xlab = "Number of clusters", ylab = "Measure of Compactness")
  return(p)
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

f_merge <- function(s_split, sepp = "")
{
  #Merges a splitted sequence into a continuous string
  seq <- NULL
  for (i in 1:length(s_split))
  {
    seq <- paste(seq, s_split[i], sep=sepp)
  }
  return (seq)
}

## Functions for comparison algos.
f_remove_seq_padding <- function(all_simulated_sequences_with_hyphen)
{
  # When we coded for simulating data, we padded with asterisks for no good reason apparently. They were separated with a hyphen. In this function we will just get a vector of sequences without any padding.
  f_split_tmp <- function(sim_seq)
  {
    if(length(grep("-",sim_seq)))
    {
      tmp     <- strsplit(sim_seq, "-")
      tmp     <- tmp[[1]][1]
    }else{
      tmp <- sim_seq
    }
    return(tmp)
  }
  sequences <- lapply(all_simulated_sequences_with_hyphen, f_split_tmp)
  return(sequences)
}

f_convert_to_fasta <- function(sequences, filename = 'sequences.fasta')
{
  library(seqRFLP)
  seqs <- f_remove_seq_padding(all_simulated_sequences_with_hyphen = sequences)
  seqs <- unlist(seqs)
  snames <- paste("s", 1:length(seqs), sep="")
  df <- data.frame(names = snames, seq = seqs)
  dd <- dataframe2fas(df, file = filename)
}

f_MUSCLE_MSA_diff_mat <- function(muscle_msa_file)
{
  # Takes the Multiple sequence assignments from MUSCLE as input and creates a pairwise distance matrix
  library(bios2mds)
  sequences.msa <- import.fasta(muscle_msa_file)
  diff_mat      <- mat.dif(sequences.msa, sequences.msa)
  return(diff_mat)
}

f_MUSCLE_MSA_clustering <- function(muscle_msa_file, nclust, do.plot = TRUE)
{
  # This function will cluster multiple sequence alignments found from MUSCLE algorithm (online)
  
  diff_mat      <- f_MUSCLE_MSA_diff_mat(muscle_msa_file)
  p             <- f_plot(diff_mat = diff_mat, chart_title = "C")
  print(p)
  class <- f_clustering(diff_mat = diff_mat, nclust = nclust)
  
  seq <- NULL
  for(i in 1:nrow(class))
  {
    print(i)
    seq.tmp <- strtoi(strsplit(class[i,'sequence'], "s")[[1]][2])
    seq <- c(seq, seq.tmp)
  }
  class[,'sequence'] <- seq
  class <- class[order(strtoi(class[,'sequence'])), ]  
  return(class)
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
    possibilities <- permutations(K,K,letters[1:K])
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
        # tmp_f1 <- 2*(out_cc$byClass["Pos Pred Value"]*out_cc$byClass["Sensitivity"]/(out_cc$byClass["Pos Pred Value"]+out_cc$byClass["Sensitivity"]))
        tmp_f1 <- f_get_f1(confusion = tmp$table)
        flag <- (tmp_f1 > out_f1)
      }else if(type == "accuracy"){
        flag <- (tmp$overall["Accuracy"] > out_cc$overall["Accuracy"])
      }
      
      # if(tmp$overall["Accuracy"] > out_cc$overall["Accuracy"]){
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


############# Functions for param selection ####################
f_best_param_given_cluster <- function(num_cluster, all_seq_mats, Moment, inv.powered = T, lambda, length_normalize = T, index_type = "db", clustering_method = "complete", numPCs = NULL, trace = T)
{
  ## This function will find the best SGT parameter, given the number of clusters
  index <- matrix(0, ncol = Moment)
  for(m in 1:Moment)
  {
    if(trace){paste("Doing moment",m,"in",Moment)}
    input_data <- f_create_input_kmeans(all_seq_mats = all_seq_mats, length_normalize = length_normalize, lambda = lambda, moment = m, trace = T, inv.powered = inv.powered)
    
    if(!is.null(numPCs))
    {
      tmp <- f_pcs(input_data = input_data, PCs = numPCs)
      input_data <- tmp$input_data_pcs
    }
    
    clusting   <- NbClust(data = input_data, distance = "manhattan",
                          min.nc=num_cluster, max.nc=num_cluster, method = clustering_method,
                          index = index_type, alphaBeale = 0.1)
    clusting
    index[1,m] <- clusting$All.index[1]
  }
  out <- list(index = index, best.param = which.min(index[1, ])) ## The best moment
  return(out)
}

f_best_cluster_given_param <- function(m, all_seq_mats, min.nc, max.nc, length_normalize = T, lambda, inv.powered =T, index_type ="db", clustering_method = "complete", trace = T)
{
  input_data <- f_create_input_kmeans(all_seq_mats = all_seq_mats, length_normalize = length_normalize, lambda = lambda, moment = m, trace = F, inv.powered = inv.powered)  
  
  if(trace){"Working on trying clusters..."}
  clusting   <- NbClust(data = input_data, distance = "manhattan",
                        min.nc=min.nc, max.nc=max.nc, method = clustering_method,
                        index = index_type, alphaBeale = 0.1)
  out <- list(clusting_result = clusting, best.nc = clusting$Best.nc[1])
  return(out)
}
