```{r }
library(ggplot2)
library(ggridges)
library(gridExtra)
library(dplyr)
library(grid)
```

```{r Load and filter data}
# NOTE: The input file 'blr_metrics_estimate.txt' is an output extracted from the normative modeling analysis

# Read the tab-separated text file containing metrics for all ROIs
df <- read.table(
  "C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/distrib_plots/blr_metrics_estimate.txt",
  header = TRUE, sep = "\t"
)

# Keep only rows corresponding to ROIs ending in 'width'
df_width <- df %>% filter(grepl("width$", ROI))

# Add an identifier column for plotting
df_width$option <- "blr_metrics_estimate"
```

```{r Define metrics and plot function}
# ===========================
# DEFINE METRICS AND PLOT FUNCTION
# ===========================

# List of performance and distribution metrics to visualize
metrics_list <- c("MSLL", "EV", "SMSE", "RMSE", "Rho", "Kurtosis", "Skew")

# Function to generate a density ridge plot for a given metric
create_density_plot <- function(data, metric) {
  ggplot(data, aes(x = .data[[metric]], y = option)) +
    geom_density_ridges(
      fill = "#1b9e77", 
      color = "black", 
      quantile_lines = TRUE, 
      quantiles = 2, 
      scale = 1
    ) +
    theme(
      panel.background = element_blank(),
      plot.background = element_blank(),
      panel.grid = element_blank(),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.title.x = element_text(size = 15),
      axis.text.x = element_text(size = 13),
      plot.title = element_blank(),
      legend.position = "none"
    ) +
    scale_y_discrete(expand = c(0, 0)) +
    coord_cartesian(clip = "off") +
    labs(x = metric)
}
```

```{r Generate density plots}
# Create an empty list to store plots
plots <- list()

# Loop over each metric and generate a corresponding density plot
for (metric in metrics_list) {
  plots[[length(plots) + 1]] <- create_density_plot(df_width, metric)
}

# If the total number of plots is odd, add a blank plot to complete the grid
if (length(plots) %% 2 != 0) {
  plots[[length(plots) + 1]] <- nullGrob()
}

```

```{r Show and export final plot grid}
# Show the plot
grid.arrange(grobs = plots, nrow = 4, ncol = 2)

# Save the final image as a with 4 rows and 2 columns
png("C:/Users/Usuario/Desktop/Practicas/AAA_80clinical/distrib_plots/distribution_grid.png", 
    width = 2400, height = 3000, res = 300)

# Arrange all plots in a grid layout
grid.arrange(grobs = plots, nrow = 4, ncol = 2)

dev.off()
```

