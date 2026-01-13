```{r}
library(dplyr)
library(ggplot2)
library(readr)
```

```{r}
train_data <- read_csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/train_data.csv")
test_data  <- read_csv("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/test_data.csv")

sulco_names <- c(
  "Left Lateral Fissure", "Right Lateral Fissure", 
  "Left Anterior Cingulate Sulcus", "Right Anterior Cingulate Sulcus", 
  "Left Posterior Cingulate Sulcus", "Right Posterior Cingulate Sulcus", 
  "Left Calcarine Fissure", "Right Calcarine Fissure", 
  "Left Collateral Fissure", "Right Collateral Fissure", 
  "Left Intra-Parietal Fissure", "Right Intra-Parietal Fissure", 
  "Left Parieto-Occipital Fissure", "Left Occipital Sulcus", 
  "Right Occipital Sulcus", "Left Central Sulcus", 
  "Right Central Sulcus", "Left Inferior Frontal Sulcus", 
  "Right Inferior Frontal Sulcus", "Left Paracingulate Sulcus", 
  "Right Paracingulate Sulcus", "Left Intermediate Frontal Sulcus", 
  "Right Intermediate Frontal Sulcus", "Left Superior Frontal Sulcus", 
  "Right Superior Frontal Sulcus", "Left Occipital-Temporal Lateral Sulcus",
  "Right Occipital-Temporal Lateral Sulcus", "Left Orbitofrontal Sulcus", 
  "Right Orbitofrontal Sulcus", "Left Medial Parietal Sulcus", 
  "Right Medial Parietal Sulcus", "Left Pre-Central Sulcus", 
  "Right Pre-Central Sulcus", "Left Post-Central Sulcus", 
  "Right Post-Central Sulcus", "Left Inferior Temporal Sulcus", 
  "Right Inferior Temporal Sulcus", "Left Superior Temporal Sulcus", 
  "Right Superior Temporal Sulcus", "Right Parieto-Occipital Fissure"
)
```

```{r Function to calculate the unexplained variance}
calculate_unexplained <- function(data, width_col, thickness_col) {
  if (width_col %in% colnames(data) & thickness_col %in% colnames(data)) {
    model <- lm(data[[width_col]] ~ data[[thickness_col]], data = data)
    r2 <- summary(model)$r.squared
    unexplained <- 1 - r2
    return(unexplained)
  }
  return(NA)
}
```

```{r Get unexplained variance of sulcal width and thickness from joined train and test sets}

combined_data <- bind_rows(train_data, test_data)

results_joined <- data.frame(Sulcus = character(), Unexplained = numeric(), stringsAsFactors = FALSE)

for (sulcus_joined in sulco_names) {
  # Convert sulcus name to column format
  sulcus_base <- gsub(" ", "_", sulcus_joined)
  sulcus_base <- gsub("-", ".", sulcus_base)
  
  sulcus_width     <- paste0(sulcus_base, "_width")
  sulcus_thickness <- paste0(sulcus_base, "_thickness")
  
  unexplained_joined <- calculate_unexplained(combined_data, sulcus_width, sulcus_thickness)
  
  results_joined <- rbind(results_joined, data.frame(Sulcus = sulcus_joined, Unexplained = unexplained_joined))
}
```

```{r Save dataframe}
write_csv(results_joined,"C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/unexplained_variance_combined.csv")
```

```{r Get unexplained variance of sulcal width and thickness separately for train and test sets}

unexplained_train <- data.frame(Sulcus = character(), Unexplained_variance = numeric(), stringsAsFactors = FALSE)
unexplained_test  <- data.frame(Sulcus = character(), Unexplained_variance = numeric(), stringsAsFactors = FALSE)

for (sulcus in sulco_names) {
  
  # Format sulcus name to match column naming convention
  sulcus_base <- gsub(" ", "_", sulcus)
  sulcus_base <- gsub("-", ".", sulcus_base)
  
  # Build column names
  sulcus_width     <- paste0(sulcus_base, "_width")
  sulcus_thickness <- paste0(sulcus_base, "_thickness")
  
  # Compute unexplained variance and label the sulcus name with "- Train" or "- Test"
  uv_train <- calculate_unexplained(train_data, sulcus_width, sulcus_thickness)
  uv_test  <- calculate_unexplained(test_data, sulcus_width, sulcus_thickness)
  
  unexplained_train <- rbind(unexplained_train, data.frame(Sulcus = paste0(sulcus, " - Train"), Unexplained_variance = uv_train))
  unexplained_test  <- rbind(unexplained_test,  data.frame(Sulcus = paste0(sulcus, " - Test"),  Unexplained_variance = uv_test))
}

# Combine results from both sets
results <- rbind(unexplained_train, unexplained_test)

```

```{r Save dataframe}
write_csv(results, "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/unexplained_var_train_test.csv")
```

