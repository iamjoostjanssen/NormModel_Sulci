```{r Load required libraries}
library(ggplot2)
library(effsize)
```

```{r Read z-score data from CSV file}
input_path <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/t-test/test_width_zscores.csv"
data <- read.csv(input_path, header = TRUE, stringsAsFactors = FALSE)
```

```{r Prepare sulcal data}
# Remove non-sulcus columns 
sulci_names <- setdiff(names(data), c("ID", "dcode", "site", "scanner"))

# Define the desired order of sulci and add "Z_" prefix
sulci_order <- c(
  "Left_Lateral_Fissure", "Right_Lateral_Fissure",
  "Left_Anterior_Cingulate_Sulcus", "Right_Anterior_Cingulate_Sulcus",
  "Left_Posterior_Cingulate_Sulcus", "Right_Posterior_Cingulate_Sulcus",
  "Left_Calcarine_Fissure", "Right_Calcarine_Fissure",
  "Left_Collateral_Fissure", "Right_Collateral_Fissure",
  "Left_Intra.Parietal_Fissure", "Right_Intra.Parietal_Fissure",
  "Left_Parieto.Occipital_Fissure", "Left_Occipital_Sulcus", "Right_Occipital_Sulcus",
  "Left_Central_Sulcus", "Right_Central_Sulcus",
  "Left_Inferior_Frontal_Sulcus", "Right_Inferior_Frontal_Sulcus",
  "Left_Paracingulate_Sulcus", "Right_Paracingulate_Sulcus",
  "Left_Intermediate_Frontal_Sulcus", "Right_Intermediate_Frontal_Sulcus",
  "Left_Superior_Frontal_Sulcus", "Right_Superior_Frontal_Sulcus",
  "Left_Occipital.Temporal_Lateral_Sulcus", "Right_Occipital.Temporal_Lateral_Sulcus",
  "Left_Orbitofrontal_Sulcus", "Right_Orbitofrontal_Sulcus",
  "Left_Medial_Parietal_Sulcus", "Right_Medial_Parietal_Sulcus",
  "Left_Pre.Central_Sulcus", "Right_Pre.Central_Sulcus",
  "Left_Post.Central_Sulcus", "Right_Post.Central_Sulcus",
  "Left_Inferior_Temporal_Sulcus", "Right_Inferior_Temporal_Sulcus",
  "Left_Superior_Temporal_Sulcus", "Right_Superior_Temporal_Sulcus",
  "Right_Parieto.Occipital_Fissure"
)
sulci_order <- paste0("Z_", sulci_order)

# Keep only those that are actually in the dataset
sulci_names <- intersect(sulci_order, sulci_names)

```

```{r Run t-tests and compute effect sizes for each sulcus}
p_values <- numeric(length(sulci_names))
t_values <- numeric(length(sulci_names))
cohens_d <- numeric(length(sulci_names))

for (i in seq_along(sulci_names)) {
  col <- sulci_names[i]
  group0 <- data[data$dcode == 0, col]
  group1 <- data[data$dcode == 1, col]
  
  t_res <- t.test(group0, group1)
  p_values[i] <- t_res$p.value
  t_values[i] <- t_res$statistic
  cohens_d[i] <- cohen.d(group0, group1)$estimate
}
```


```{r Apply False Discovery Rate correction to p-values}
fdr_p <- p.adjust(p_values, method = "fdr")
```

```{r Create results dataFrame}
results <- data.frame(
  sulcus = sulci_names,
  p_value = p_values,
  t_value = t_values,
  fdr_p_value = fdr_p,
  cohens_d = cohens_d,
  stringsAsFactors = FALSE
)

# Format sulcus names for plotting
plot_names <- gsub("^Z_", "", results$sulcus)
plot_names <- gsub("_", " ", plot_names)
plot_names <- gsub("\\.", "-", plot_names)
plot_names <- gsub(" width$", "", plot_names, ignore.case = TRUE)
plot_names <- trimws(plot_names)

results$Sulcus <- factor(plot_names, levels = plot_names)
```

```{r Save results to CSV}
write.csv(results, "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/t-test/t_test_width.csv", row.names = FALSE)

```

```{r Plot p-values}
ggplot(results, aes(x = Sulcus, y = p_value)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "P-values per Sulcus Width", x = "Sulcus", y = "P-value")

```
```{r Plot FDR-corrected P-values}
ggplot(results, aes(x = Sulcus, y = fdr_p_value)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "FDR-corrected P-values per Sulcus Width", x = "Sulcus", y = "FDR P-value")

```

```{r Plot T-values}
ggplot(results, aes(x = Sulcus, y = t_value)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "T-values per Sulcus Width", x = "Sulcus", y = "T-value")

```


```{r Plot Cohen's d}
ggplot(results, aes(x = Sulcus, y = cohens_d)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Cohen's d per Sulcus Width", x = "Sulcus", y = "Cohen's d")

```
