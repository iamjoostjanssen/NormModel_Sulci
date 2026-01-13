```{r }
# WARNING: Sulcus order must match the following exactly: 
# "Left Central Sulcus", "Left Pre-Central Sulcus", 
# "Left Post-Central Sulcus", "Left Intra-Parietal Fissure",
# "Left Superior Frontal Sulcus", "Left Parieto-Occipital Fissure", 
# "Left Posterior Cingulate Sulcus", "Left Paracingulate Sulcus", 
# "Left Inferior Frontal Sulcus", "Left Orbitofrontal Sulcus", 
# "Left Lateral Fissure", "Left Anterior Cingulate Sulcus", 
# "Left Superior Temporal Sulcus", "Left Occipital Sulcus", 
# "Left Collateral Fissure", "Left Medial Parietal Sulcus", 
# "Left Intermediate Frontal Sulcus", "Left Inferior Temporal Sulcus", 
# "Left Occipital-Temporal Lateral Sulcus", "Left Calcarine Fissure",
# "Right Central Sulcus", "Right Pre-Central Sulcus", 
# "Right Post-Central Sulcus", "Right Intra-Parietal Fissure",
# "Right Superior Frontal Sulcus", "Right Parieto-Occipital Fissure", 
# "Right Posterior Cingulate Sulcus", "Right Paracingulate Sulcus", 
# "Right Inferior Frontal Sulcus", "Right Orbitofrontal Sulcus", 
# "Right Lateral Fissure", "Right Anterior Cingulate Sulcus", 
# "Right Superior Temporal Sulcus", "Right Occipital Sulcus", 
# "Right Collateral Fissure", "Right Medial Parietal Sulcus", 
# "Right Intermediate Frontal Sulcus", "Right Inferior Temporal Sulcus", 
# "Right Occipital-Temporal Lateral Sulcus", "Right Calcarine Fissure"
```

```{r }
if (!require("matrixStats")) install.packages("matrixStats")
if (!require("clue")) install.packages("clue")

library(matrixStats)  # For rowMins() used in 'vasa' method
library(clue)         # For solve_LSAP(), implementing the Hungarian algorithm
library(ggplot2)
library(dplyr)
```

```{r Load sulcal coordinate data for left hemisphere}
LH_data <- data.frame(
  sulcus = c("Central Sulcus", "Pre-Central Sulcus", 
             "Post-Central Sulcus", "Intra-Parietal Fissure",
             "Superior Frontal Sulcus", "Parieto-Occipital Fissure", 
             "Posterior Cingulate Sulcus", "Paracingulate Sulcus", 
             "Inferior Frontal Sulcus", "Orbitofrontal Sulcus", 
             "Lateral Fissure", "Anterior Cingulate Sulcus", 
             "Superior Temporal Sulcus", "Occipital Sulcus", 
             "Collateral Fissure", "Medial Parietal Sulcus", 
             "Intermediate Frontal Sulcus", "Inferior Temporal Sulcus",
             "Occipital-Temporal Lateral Sulcus", "Calcarine Fissure"),
  LHx = c(-37.3, -36.8, -41.1, -28.4, -23.7, -12.7, -11.7, -7.5, -41.1, -26.7, 
          -44.6, -10.0, -50.5, -33.8, -28.1, -8.4, -29.9, -51.0, -41.2, -13.3),
  LHy = c(-19.8, -2.9, -33.0, -65.6, 19.6, -70.0, -34.9, 33.5, 26.7, 36.3, 
          -8.3, 23.3, -41.7, -85.5, -53.1, -54.6, 42.8, -28.4, -38.4, -70.9),
  LHz = c(48.2, 43.3, 46.8, 38.3, 46.0, 26.3, 51.1, 30.6, 19.8, -14.9, 
          6.0, 21.1, 5.7, 1.5, -13.0, 40.1, 17.2, -21.1, -22.4, 3.3)
)

# Load sulcal coordinate data for right hemisphere
RH_data <- data.frame(
  sulcus = c("Central Sulcus", "Pre-Central Sulcus", 
             "Post-Central Sulcus", "Intra-Parietal Fissure",
             "Superior Frontal Sulcus", "Parieto-Occipital Fissure", 
             "Posterior Cingulate Sulcus", "Paracingulate Sulcus", 
             "Inferior Frontal Sulcus", "Orbitofrontal Sulcus", 
             "Lateral Fissure", "Anterior Cingulate Sulcus", 
             "Superior Temporal Sulcus", "Occipital Sulcus", 
             "Collateral Fissure", "Medial Parietal Sulcus", 
             "Intermediate Frontal Sulcus", "Inferior Temporal Sulcus",
             "Occipital-Temporal Lateral Sulcus", "Calcarine Fissure"),
  RHx = c(36.2, 36.1, 39.5, 30.0, 24.2, 14.4, 11.7, 8.4, 41.7, 26.9, 
          26.8, 11.1, 51.0, 34.9, 29.3, 8.7, 30.1, 52.2, 42.1, 15.3),
  RHy = c(-18.8, -2.6, -31.4, -62.5, 17.6, -68.5, -35.2, 33.7, 27.2, 37.4, 
          11.2, 23.6, -39.3, -83.3, -53.0, -54.2, 41.2, -27.9, -37.5, -68.6),
  RHz = c(48.4, 44.2, 48.2, 39.5, 47.4, 27.8, 51.2, 30.5, 19.7, -14.1, 
          -16.6, 22.4, 7.6, 3.7, -12.6, 40.4, 19.3, -19.3, -21.7, 3.7)
)

# Prefix sulcus names
LH_data$sulcus <- ifelse(grepl("^Left", LH_data$sulcus), LH_data$sulcus, paste("Left", LH_data$sulcus))
RH_data$sulcus <- ifelse(grepl("^Right", RH_data$sulcus), RH_data$sulcus, paste("Right", RH_data$sulcus))

# Rename columns and combine hemispheres
colnames(LH_data) <- c("sulcus", "x", "y", "z")
colnames(RH_data) <- c("sulcus", "x", "y", "z")
combined_data <- rbind(LH_data, RH_data)

# Define sulcus order
sulco_order <- c( "Left Central Sulcus", "Left Pre-Central Sulcus", 
                 "Left Post-Central Sulcus", "Left Intra-Parietal Fissure",
                 "Left Superior Frontal Sulcus", "Left Parieto-Occipital Fissure", 
                 "Left Posterior Cingulate Sulcus", "Left Paracingulate Sulcus", 
                 "Left Inferior Frontal Sulcus", "Left Orbitofrontal Sulcus", 
                 "Left Lateral Fissure", "Left Anterior Cingulate Sulcus", 
                 "Left Superior Temporal Sulcus", "Left Occipital Sulcus", 
                 "Left Collateral Fissure", "Left Medial Parietal Sulcus", 
                 "Left Intermediate Frontal Sulcus", "Left Inferior Temporal Sulcus", 
                 "Left Occipital-Temporal Lateral Sulcus", "Left Calcarine Fissure",
                 "Right Central Sulcus", "Right Pre-Central Sulcus", 
                 "Right Post-Central Sulcus", "Right Intra-Parietal Fissure",
                 "Right Superior Frontal Sulcus", "Right Parieto-Occipital Fissure", 
                 "Right Posterior Cingulate Sulcus", "Right Paracingulate Sulcus", 
                 "Right Inferior Frontal Sulcus", "Right Orbitofrontal Sulcus", 
                 "Right Lateral Fissure", "Right Anterior Cingulate Sulcus", 
                 "Right Superior Temporal Sulcus", "Right Occipital Sulcus", 
                 "Right Collateral Fissure", "Right Medial Parietal Sulcus", 
                 "Right Intermediate Frontal Sulcus", "Right Inferior Temporal Sulcus", 
                 "Right Occipital-Temporal Lateral Sulcus", "Right Calcarine Fissure"
)

combined_data <- combined_data[match(sulco_order, combined_data$sulcus),]

```

```{r Generate rotation-based null model for sulcal parcellation}

# Applies random 3D rotations to sulcal coordinates and uses the Hungarian
# algorithm to find optimal region correspondences. Generates 'nrot' valid
# permutations avoiding identity mappings, for use in spatial spin tests.

rotate.parcellation <- function(coords, nrot = 10000, method = 'hungarian') {
  if (!all(dim(coords)[2] == 3)) {
    if (all(dim(coords)[1] == 3)) {
      print('Transposing coordinates to be of dimension nROI x 3')
      coords <- t(coords)
    }
  }
  
  nroi <- dim(coords)[1]
  perm.id <- array(0, dim = c(nroi, nrot))
  r <- 0
  c <- 0
  
  I1 <- diag(3)
  I1[1, 1] <- -1
  
  while (r < nrot) {
    A <- matrix(rnorm(9), 3, 3)
    qrdec <- qr(A)
    TL <- qr.Q(qrdec)
    temp <- qr.R(qrdec)
    TL <- TL %*% diag(sign(diag(temp)))
    if (det(TL) < 0) TL[, 1] <- -TL[, 1]
    TR <- I1 %*% TL %*% I1
    coords.rot <- coords %*% TL
    
    dist <- matrix(0, nroi, nroi)
    for (i in 1:nroi) {
      for (j in 1:nroi) {
        dist[i, j] <- sqrt(sum((coords[i, ] - coords.rot[j, ])^2))
      }
    }
    
    if (method == 'vasa') {
      temp.dist <- dist
      rot <- ref <- integer()
      for (i in 1:nroi) {
        ref.ix <- which(rowMins(temp.dist, na.rm = TRUE) == max(rowMins(temp.dist, na.rm = TRUE)))
        rot.ix <- which(temp.dist[ref.ix, ] == min(temp.dist[ref.ix, ], na.rm = TRUE))
        ref <- c(ref, ref.ix)
        rot <- c(rot, rot.ix)
        temp.dist[, rot.ix] <- NA
        temp.dist[ref.ix, ] <- 0
      }
    } else if (method == 'hungarian') {
      rot <- solve_LSAP(dist, maximum = FALSE)
      ref <- 1:nroi
    } else {
      stop("Invalid method. Use 'vasa' or 'hungarian'.")
    }
    
    if (!all(sort(rot) == 1:nroi)) stop("Permutation error")
    
    if (!all(rot == 1:nroi)) {
      r <- r + 1
      perm.id[, r] <- rot
    } else {
      c <- c + 1
    }
  }
  
  return(perm.id)
}

# Convert coordinate data to matrix and generate permutations
coords <- as.matrix(combined_data[, c("x", "y", "z")])
perm <- rotate.parcellation(coords, nrot = 1000, method = "hungarian")
```

```{r Compute spin test p-value from permuted coordinates}

# Given empirical vectors `x` and `y`, and a rotation-based null model (`perm.id`),
# this function computes a spin-test p-value by comparing the observed Spearman
# correlation to a distribution of correlations from rotated parcellations.
# It returns the average of the two directional p-values (x|y and y|x).

perm.sphere.p <- function(x, y, perm.id, corr.type = 'spearman') {
  nroi <- dim(perm.id)[1]
  nperm <- dim(perm.id)[2]
  rho.emp <- cor(x, y, method = corr.type)
  
  x.perm <- y.perm <- array(NA, dim = c(nroi, nperm))
  for (r in 1:nperm) {
    for (i in 1:nroi) {
      x.perm[i, r] <- x[perm.id[i, r]]
      y.perm[i, r] <- y[perm.id[i, r]]
    }
  }
  
  rho.null.xy <- rho.null.yx <- numeric(nperm)
  for (r in 1:nperm) {
    rho.null.xy[r] <- cor(x.perm[, r], y, method = corr.type)
    rho.null.yx[r] <- cor(y.perm[, r], x, method = corr.type)
  }
  
  if (rho.emp > 0) {
    p.perm.xy <- sum(rho.null.xy > rho.emp) / nperm
    p.perm.yx <- sum(rho.null.yx > rho.emp) / nperm
  } else {
    p.perm.xy <- sum(rho.null.xy < rho.emp) / nperm
    p.perm.yx <- sum(rho.null.yx < rho.emp) / nperm
  }
  
  return((p.perm.xy + p.perm.yx) / 2)
}
```

```{r Group-Level Hub Vulnerability}

# --- Description ---
# This analysis evaluates whether sulci that are more structurally connected in the normative population (i.e., hubs) show greater morphometric alterations in schizophrenia. Specifically, it tests the hypothesis that sulci with higher weighted degree centrality are more affected by the disorder.
# The test computes the Spearman correlation between the group-level sulcal t-map and the normative weighted degree vector. A spatial spin permutation test assesses the significance of the observed correlation.


# --- Load t-values or Cohen's d vector ---
t_map <- read.csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/t-test/reordered_all_tests_width.csv", stringsAsFactors = FALSE)
t_values <- -t_map$t_value

# --- Load centrality values (Weighted Degree only) ---
centrality_df <- read.csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Epicenter_mapping/threshold_0/igraph/centralities_nfiber.csv",stringsAsFactors = FALSE)
weighted_degree <- centrality_df$Weighted_Degree  

# --- Compute empirical correlation ---
empirical_rho <- cor(t_values, weighted_degree, method = "spearman")

# --- Compute spin test p-value ---
spin_p <- perm.sphere.p(x = t_values,
                        y = weighted_degree,
                        perm.id = perm,
                        corr.type = "spearman")

# --- Output results ---
cat("Group-level Spearman correlation:",round(empirical_rho, 3), "\n")
cat("Spin test p-value:", round(spin_p, 4), "\n")
```

```{r Group-Level Epicenter Mapping}

# --- Description ---
# This analysis identifies potential epicenters of disease-related alterations at the group level.
# For each sulcus, it computes the Spearman correlation between its normative connectivity fingerprint (i.e., its connections to all other sulci) and the group-level t-map.
# Sulci whose connectivity profiles significantly align with the observed pattern of group-level abnormalities are considered candidate epicenters. Spatial spin permutation tests are used to assess the statistica significance.


# --- Load t-values or Cohen's d vector ---
t_map <- read.csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/t-test/reordered_all_tests_width.csv", stringsAsFactors = FALSE)
t_values <- -t_map$t_value

# --- Load normative connectivity matrix ---
connectivity_file <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Epicenter_mapping/threshold_0/new_matrices/new_nfiber.csv"
connectivity_df <- read.csv(connectivity_file, stringsAsFactors = FALSE)

# --- Load reference morphometric map (e.g., t-values or centrality) ---
reference_file <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Epicenter_mapping/threshold_0/igraph/centralities_nfiber.csv"
reference_df <- read.csv(reference_file, stringsAsFactors = FALSE)
reference_vector <- reference_df$Weighted_Degree  # Vector of reference values (length = n sulci)

# --- Initialize result matrix ---
n_sulci <- ncol(connectivity_df)
group_rho_matrix <- matrix(NA, nrow = n_sulci, ncol = 3)
colnames(group_rho_matrix) <- c("Sulcus", "Spearman_Correlation", "P_Spin")

# --- Loop over sulci: compare each fingerprint to reference vector ---
for (s in 1:n_sulci) {
  fingerprint_sulcus <- connectivity_df[[s]]  # Connectivity profile of sulcus s
  
  # Compute Spearman correlation with reference map
  rho_val <- cor(t_values, fingerprint_sulcus, method = "spearman")
  
  # Compute spin-based p-value
  p_spin_val <- perm.sphere.p(
    x = t_values,
    y = fingerprint_sulcus,
    perm.id = perm,
    corr.type = "spearman"
  )
  
  # Store results
  group_rho_matrix[s, ] <- c(sulco_order[s], rho_val, p_spin_val)
}

# --- Convert results to data frame ---
group_epicenter_df <- as.data.frame(group_rho_matrix, stringsAsFactors = FALSE)
colnames(group_epicenter_df) <- c("Sulcus", "Spearman_Correlation", "P_Spin")
group_epicenter_df$Spearman_Correlation <- as.numeric(group_epicenter_df$Spearman_Correlation)
group_epicenter_df$P_Spin <- as.numeric(group_epicenter_df$P_Spin)

# --- Save results to file ---
output_dir_group <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Epicenter_mapping/threshold_0/spin_nfiber_weighteddegree/"
write.csv(group_epicenter_df,
          file = file.path(output_dir_group, "group_epicenter_mapping.csv"),
          row.names = FALSE)
```

```{r Individual-Level Hub Vulnerability Test}

# --- Description ---
# This analysis examines whether the pattern of individual morphometric deviations in each patient is aligned with the normative hubness architecture of the brain.
# For each subject, it computes the Spearman correlation between the subject's sulcal z-score map (quantifying deviations from the normative model) and the normative weighted degree vector.
# A spatial spin test is used to assess the statistical significance of this correlation.
# The proportion of patients with significantly positive correlations is tested against chance using a binomial test.


# --- Load subject-level z-scores ---
zscores_hub <- read.csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/z_scores/test_width_zscores.csv",
                        header = TRUE, stringsAsFactors = FALSE)

if ("dcode" %in% colnames(zscores_hub)) {
  zscores_hub <- subset(zscores_hub, dcode == 1)
  zscores_hub <- zscores_hub[, !(names(zscores_hub) %in% c("dcode", "site", "scanner"))]
}

# Standardize column names
colnames(zscores_hub)[-1] <- gsub("^Z_", "", colnames(zscores_hub)[-1])
colnames(zscores_hub)[-1] <- gsub("_", " ", colnames(zscores_hub)[-1])
colnames(zscores_hub)[-1] <- gsub("\\.", "-", colnames(zscores_hub)[-1])

# Ensure numeric data and correct sulcus order
zscores_hub[, -1] <- lapply(zscores_hub[, -1], as.numeric)
sulci_names_hub <- colnames(zscores_hub)[-1]
reordered_indices <- match(sulco_order, sulci_names_hub)
reordered_indices <- reordered_indices[!is.na(reordered_indices)]
zscores_hub <- zscores_hub[, c(1, reordered_indices + 1)]

# --- Load centrality vector ---
centrality_df <- read.csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Epicenter_mapping/threshold_1/igraph/centralities_nfiber.csv",
                          stringsAsFactors = FALSE)
centrality_vector <- centrality_df$Weighted_Degree[match(sulco_order, centrality_df$sulcus)]

# --- Compute correlation and spin p-value per subject ---
n_subjects <- nrow(zscores_hub)
hub_results <- data.frame(ID = zscores_hub$ID, rho = NA, p_spin = NA)

for (i in 1:n_subjects) {
  z_patient <- as.numeric(zscores_hub[i, -1])
  rho <- cor(centrality_vector, z_patient, method = "spearman")
  p_val <- perm.sphere.p(x = centrality_vector, y = z_patient, perm.id = perm, corr.type = "spearman")
  hub_results$rho[i] <- rho
  hub_results$p_spin[i] <- p_val
}

print(hub_results)

# --- Summarize proportion of patients with significant positive correlation ---
sig_patients <- subset(hub_results, p_spin < 0.05 & rho > 0)
prop_sig <- nrow(sig_patients) / n_subjects * 100

cat("Number of patients:", n_subjects, "\n")
cat("Significant & positive:", nrow(sig_patients), "\n")
cat("Percentage:", round(prop_sig, 2), "%\n")

# --- Chi-square test against 2.5% null ---
prop_test <- prop.test(x = nrow(sig_patients), n = n_subjects, p = 0.025, alternative = "greater")
print(prop_test)
```
