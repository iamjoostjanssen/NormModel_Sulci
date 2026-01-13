```{r}
library(tibble)
library(dplyr)
library(caret)
```


```{r Load and filter data}
# Read the full sulcal dataset
megasample_r <- read.csv("C:/Users/Usuario/Desktop/Practicas/sulcal_data.csv", sep = ",")

# Extract unique scanning sites
sites <- unique(megasample_r$site)

# Define control and clinical sites
sites_controls <- c('aomic_id1000', 'aomic_piop1', 'aomic_piop2', 'camcan', 'dlbs', 'ixi', 'narratives', 'oasis', 'rockland', 'sald')
sites_clinic_SCZ <- c('LA5C_study', 'SRPBS_open', 'bgs', 'cobre', 'utrecht')

# Filter the dataset to obtain separate groups
controls <- megasample_r[megasample_r$site %in% sites_controls, ]
clinic_SCZ <- megasample_r[megasample_r$site %in% sites_clinic_SCZ, ]

# Separate clinical controls and patients
controls_clinic_SCZ <- clinic_SCZ[clinic_SCZ$dcode == 0, ]
patients_clinic_SCZ <- clinic_SCZ[clinic_SCZ$dcode == 1, ]

# Create final patient and clinical control dataframes
patients <- rbind(patients_clinic_SCZ)
control_clinic <- rbind(controls_clinic_SCZ)

# Check available timepoints in the dataset
unique(megasample_r$timepoint)

```

```{r Partitioning function}
# Function to create stratification groups based on numeric/categorical variables
createGroups <- function (data, numericGroups = 3) {
  result <- NULL
  for (column in colnames(data)) {
    tmp <- data[, column]
    if (is.numeric(tmp)) {
      tmp <- cut(tmp,
                 unique(quantile(tmp, probs = seq(0, 1, length = numericGroups))),
                 include.lowest = TRUE)
    }
    if (is.null(result)) {
      result <- tmp
    } else {
      result <- paste0(result, tmp)
    }
  }
  result
}
```

```{r Non-clinical healthy controls split}
set.seed(2)  # Ensure reproducibility

# Keep only baseline data (timepoint 1) from clinical controls
control_clinic_baseline <- control_clinic[control_clinic$timepoint == 1, ]

df <- controls  # Select full healthy control sample
data <- df[, c("age", "sex", "scanner")]  # Variables for stratified split

str(data)  # Inspect structure
table(data$scanner)  # Check scanner distribution

# Create stratified groups
groups <- createGroups(data)

# Split 80% train / 20% test
index <- createDataPartition(groups, p = 0.80, list = FALSE)
split1 <- df[index, ]  # Train set
split2 <- df[-index, ]  # Test set

# Plot age density comparison
ggplot() +
  geom_density(data = split1, aes(age)) +
  geom_density(data = split2, aes(age))

# Compare sex distribution in train/test
a <- table(split1$sex)
(a[1] / (a[1] + a[2])) * 100

a <- table(split2$sex)
(a[1] / (a[1] + a[2])) * 100

# Compare scanner distribution (scanner type 4)
(table(split1$scanner)[4] / nrow(split1)) * 100
(table(split2$scanner)[4] / nrow(split2)) * 100

# Density plot with color differentiation
ggplot() +
  geom_density(data = split1, aes(age, color = "Train"), linetype = "dashed") +
  geom_density(data = split2, aes(age, color = "Test"), linetype = "dashed") +
  scale_color_manual(values = c("Train" = "blue", "Test" = "red"), labels = c("Train", "Test")) +
  labs(x = "Age", y = "Density", title = "Density Plot of Age for Train and Test Sets") +
  theme_minimal()

# Save results
write.csv(split1, "C:/Users/Usuario/Documents/controls_split_80.csv", row.names = FALSE)
write.csv(split2, "C:/Users/Usuario/Documents/controls_split_20.csv", row.names = FALSE)

```

```{r Clinical healthy controls split}
df <- control_clinic_baseline
data <- df[, c("age", "sex", "scanner")]

str(data)
table(data$scanner)

# Stratified grouping and split
groups <- createGroups(data)
index <- createDataPartition(groups, p = 0.80, list = FALSE)
split1 <- df[index, ]
split2 <- df[-index, ]

# Density plot
ggplot() +
  geom_density(data = split1, aes(age, color = "Train"), linetype = "dashed") +
  geom_density(data = split2, aes(age, color = "Test"), linetype = "dashed") +
  scale_color_manual(values = c("Train" = "blue", "Test" = "red"), labels = c("Train", "Test")) +
  labs(x = "Age", y = "Density", title = "Density Plot of Age for Train and Test Sets") +
  theme_minimal()

# Compare sex percentages
a <- table(split1$sex)
(a[1] / (a[1] + a[2])) * 100

a <- table(split2$sex)
(a[1] / (a[1] + a[2])) * 100

# Compare scanner type 4
(table(split1$scanner)[4] / nrow(split1)) * 100
(table(split2$scanner)[4] / nrow(split2)) * 100

# Follow-up timepoints to baseline splits

# Timepoint 2 for training set
u_tp2 <- control_clinic[control_clinic$timepoint == 2, ]
u_tp2 <- u_tp2[u_tp2$subID %in% split1$subID, ]
control_clinic_split1 <- rbind(split1, u_tp2)

# Timepoint 2 for test set
u_tp2 <- control_clinic[control_clinic$timepoint == 2, ]
u_tp2 <- u_tp2[u_tp2$subID %in% split2$subID, ]
control_clinic_split2 <- rbind(split2, u_tp2)

# Save both splits
write.csv(control_clinic_split1, "C:/Users/Usuario/Documents/controls_clinic_split_80.csv", row.names = FALSE)
write.csv(control_clinic_split2, "C:/Users/Usuario/Documents/controls_clinic_split_20.csv", row.names = FALSE)
```

```{r Patients split}
# Save full patient set 
write.csv(patients, "C:/Users/Usuario/Documents/patients_clinic.csv", row.names = FALSE)
```
