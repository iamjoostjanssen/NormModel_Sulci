```{r}
library(dplyr)
library(dbscan)
```

```{r Load the data and select the variables of interest}
# Set the path to the CSV file containing sulcal features
file_path <- "C:/Users/Usuario/Desktop/Practicas/sulcal_data.csv"

# Read the dataset
data <- read.csv(file_path)

# Select columns whose names end with "_width" or "_thickness"
columnas_seleccionadas <- names(data)[
  grepl("_(width|thickness)$", names(data))
]

# Create a filtered dataframe with only selected columns
datos_filtrados <- data[, columnas_seleccionadas]
```

```{r Compute Local Outlier Factor (LOF)}
# Convert the selected columns to a numeric matrix
X <- as.matrix(datos_filtrados)

# Compute LOF scores using dbscan::lof with minPts = 10
lof_scores <- lof(X, minPts = 10)
```

```{r Identify outliers}
# Set the LOF score threshold at the 95th percentile
threshold <- quantile(lof_scores, 0.95)

# Identify rows with LOF scores above threshold
outliers <- which(lof_scores > threshold)

# Remove index 1 if it's present (possible artifact or header row)
outliers <- which(outliers > 1)

# Create a dataframe with details about the outliers
outliers_info <- data.frame(
  Row = outliers,
  LOF_Score = lof_scores[outliers],
  Metric = apply(
    datos_filtrados[outliers, , drop = FALSE], 
    1, 
    function(row) names(row)[which.max(row)]
  )
)

# Print the outlier details to the console
print(outliers_info)
```

```{r Remove outliers and save the new dataframe}

# Create a cleaned version of the original dataframe without outlier rows
outlier_adjusted_dataframe <- data[-outliers, ]

# Save the cleaned dataframe to CSV
write.csv(
  outlier_adjusted_dataframe,
  file = "C:/Users/Usuario/Desktop/Practicas/train_test_sets/outlier_adjusted_dataframe.csv",
  row.names = FALSE
)
```


