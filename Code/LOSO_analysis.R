```{r}
library(dplyr)
library(ggplot2)
library(stats)
```

```{r Define working directories}
dcode_dir <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/LOSO/dcode"
zscore_dir <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/LOSO/Z_score/width"
output_dir <- file.path(zscore_dir, "T-values")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
```

```{r List dcode files}
dcode_files <- list.files(dcode_dir, pattern = "^dcode_.*\\.csv$", full.names = TRUE)
cat("Found", length(dcode_files), "dcode files.\n")
```

```{r Compute tvalues per site}
for (file in dcode_files) {
  file_name <- basename(file)
  site <- sub("^dcode_", "", file_name)
  site <- sub("\\.csv$", "", site)
  cat("\nProcessing site:", site, "\n")

  dcode_data <- read.csv(file, header = TRUE, stringsAsFactors = FALSE)
  if (!"dcode" %in% names(dcode_data)) {
    cat("No 'dcode' column in:", file, "\n")
    next
  }

  dcode_vec <- dcode_data$dcode
  n_rows <- length(dcode_vec)
  site_zscore_dir <- file.path(zscore_dir, site)

  if (!dir.exists(site_zscore_dir)) {
    cat("Missing z-score folder for site:", site_zscore_dir, "\n")
    next
  }

  z_files <- list.files(site_zscore_dir, pattern = "^Z_predict_.*\\.txt$", full.names = TRUE)
  cat("Found", length(z_files), "Z_predict files for site:", site, "\n")

  t_results <- data.frame(Sulcus = character(), t_value = numeric(), stringsAsFactors = FALSE)

  for (z_file in z_files) {
    z_file_name <- basename(z_file)
    sulcus_name <- sub("^Z_predict_", "", z_file_name)
    sulcus_name <- sub("\\.txt$", "", sulcus_name)

    z_scores <- scan(z_file, what = numeric(), quiet = TRUE)

    if (length(z_scores) != n_rows) {
      cat("Mismatch in rows between z-scores and dcode in:", z_file, "\n")
      next
    }

    combined_data <- data.frame(dcode = dcode_vec, z_score = z_scores)
    t_test <- t.test(z_score ~ dcode, data = combined_data)

    t_results <- rbind(t_results,
                       data.frame(Sulcus = sulcus_name, t_value = t_test$statistic,
                                  stringsAsFactors = FALSE))
  }

  output_file <- file.path(output_dir, paste0(site, "_t_values.csv"))
  write.csv(t_results, output_file, row.names = FALSE)
  cat("Saved t-values to:", output_file, "\n")
}
```

```{r Combine t-values}
# Read sulcus order from a reference t-value file (e.g., from the main dataset)
ref_file <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/t-test/t_test_width.csv"
if (!file.exists(ref_file)) stop("Reference t-value file not found.")

ref_data <- read.csv(ref_file, stringsAsFactors = FALSE)
ref_data$Sulcus <- gsub(" ", "_", ref_data$Sulcus)
ref_data$Sulcus <- gsub("-", ".", ref_data$Sulcus)
ref_data$Sulcus <- paste0(ref_data$Sulcus, "_width")

ordered_sulci <- ref_data %>% arrange(t_value) %>% pull(Sulcus)

# Combine all site-specific t-values
sites <- c('aomic_id1000', 'aomic_piop1', 'aomic_piop2', 'camcan', 'dlbs', 
           'ixi', 'narratives', 'oasis', 'rockland', 'sald', 'LA5C_study', 
           'SRPBS_open', 'bgs', 'cobre', 'utrecht')

loso_data_list <- list()

for (site in sites) {
  site_file <- file.path(output_dir, paste0(site, "_t_values.csv"))
  if (!file.exists(site_file)) {
    message("Missing file:", site_file)
    next
  }
  site_data <- read.csv(site_file, stringsAsFactors = FALSE)
  if (!all(c("Sulcus", "t_value") %in% names(site_data))) next
  site_data$site <- site
  loso_data_list[[site]] <- site_data
}

all_loso_data <- do.call(rbind, loso_data_list)
```

```{r Clean order plot}
all_loso_data$Sulcus <- factor(all_loso_data$Sulcus, levels = ordered_sulci)
all_loso_data <- all_loso_data %>% arrange(Sulcus)

# Clean sulcus names for plotting
levels(all_loso_data$Sulcus) <- gsub("_width$", "", levels(all_loso_data$Sulcus))
levels(all_loso_data$Sulcus) <- gsub("_", "  ", levels(all_loso_data$Sulcus))
levels(all_loso_data$Sulcus) <- gsub("\\.", "-", levels(all_loso_data$Sulcus))

# Plot the t-values colored by site
p <- ggplot(all_loso_data, aes(x = Sulcus, y = t_value, color = site)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "LOSO Analysis of Sulcal Width Z-scores", x = "Sulcus", y = "t-value")

print(p)
```

```{r}
ggsave("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/LOSO/Z_score/width/plot_width.png",
       p, width = 10, height = 6)
```
