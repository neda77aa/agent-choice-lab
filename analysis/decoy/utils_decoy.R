library(tidyverse)
library(fixest)

create_output_dirs <- function(base_dir = "output/decoy/") {
  if (!dir.exists(base_dir)) dir.create(base_dir, recursive = TRUE)
}

save_plot <- function(plot, filename, width = 9, height = 6) {
  create_output_dirs(dirname(filename))
  ggplot2::ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    device = "pdf"
  )
}

theme_publication <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom"
    )
}

write_table <- function(df, path) {
  create_output_dirs(dirname(path))
  readr::write_csv(df, path)
}
