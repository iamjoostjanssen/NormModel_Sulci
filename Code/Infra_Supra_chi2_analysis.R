```{r }
library(dplyr)
library(tidyr)
library(rcompanion) 
library(ggplot2)
```

```{r }
base_path <- "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/Train_Test/z96/"
data <- read.csv(paste0(base_path, "test_width_zscores.csv"))

# List of sulcal regions (exclude metadata columns)
sulcus_cols <- setdiff(names(data), c("ID", "dcode", "site", "scanner"))
```

```{r Function: Classify z-scores into categories}
classify_z <- function(z) {
  if (is.na(z)) return(NA)
  if (z < -1.96) return("infra")
  if (z > 1.96) return("supra")
  return("normal")
}

# Apply classification to all sulci
classified_data <- data
classified_data[sulcus_cols] <- lapply(data[sulcus_cols], function(col) sapply(col, classify_z))

# Convert to long format for grouped analysis
df_long <- classified_data %>%
  pivot_longer(cols = all_of(sulcus_cols), names_to = "region", values_to = "z_category")
```

```{r Function: Analyze one category (infra or supra)}

analyze_category <- function(category_label) {
  results <- data.frame()
  
  for (region in unique(df_long$region)) {
    
    # Filter by sulcus and category
    sub_df <- df_long %>%
      filter(region == !!region & z_category %in% c(category_label, "normal")) %>%
      mutate(group = ifelse(dcode == 0, "HC", "SZ"),
             z_class = ifelse(z_category == category_label, "YES", "NO"))
    
    # Contingency table: HC/SZ vs YES/NO
    tab <- table(sub_df$group, sub_df$z_class)
    
    # Only analyze valid tables
    if (all(c("HC", "SZ") %in% rownames(tab)) && all(c("YES", "NO") %in% colnames(tab))) {
      test <- chisq.test(tab, correct = FALSE)
      w <- cohenW(tab)
      
      results <- rbind(results, data.frame(
        sulcus = region,
        HC_yes = tab["HC", "YES"],
        HC_no  = tab["HC", "NO"],
        SZ_yes = tab["SZ", "YES"],
        SZ_no  = tab["SZ", "NO"],
        p_value = test$p.value,
        chisq   = test$statistic,
        cohen_w = as.numeric(w)
      ))
    }
  }
  
  # Apply FDR correction
  results$p_fdr <- p.adjust(results$p_value, method = "fdr")
  
  # Sort by effect size
  results <- results %>% arrange(desc(cohen_w))
  return(results)
}
```

```{r Run chi-square test for both categories}
results_infra <- analyze_category("infra")
results_supra <- analyze_category("supra")
```

```{r Save results to CSV}
write.csv(results_infra, paste0(base_path, "infra_chi2_cohenw_results.csv"), row.names = FALSE)
write.csv(results_supra, paste0(base_path, "supra_chi2_cohenw_results.csv"), row.names = FALSE)
```

