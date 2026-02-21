source("decoy/utils_decoy.R")
source("decoy/data_prep_decoy.R")
source("decoy/models_decoy.R")
source("decoy/plots_decoy.R")

run_decoy_analysis <- function(input_files, model_labels, output_dir = "output/decoy/") {
  create_output_dirs(output_dir)

  all_data <- purrr::map2_dfr(input_files, model_labels, prepare_decoy_data)

  triad_level <- make_triad_level(all_data)
  lift_level <- make_lift_level(triad_level)

  # Core summaries
  summary_tbl <- summarize_lifts(lift_level)
  write_table(summary_tbl, file.path(output_dir, "decoy_summary_by_model.csv"))
  write_table(triad_level, file.path(output_dir, "triad_level_metrics.csv"))
  write_table(lift_level, file.path(output_dir, "triad_level_lifts.csv"))

  # Models
  mods <- fit_decoy_models(all_data)
  fixest::etable(
    list(
      "Target Share" = mods$m_target,
      "Target Share (Cond TC)" = mods$m_target_cond_tc
    ),
    file = file.path(output_dir, "decoy_model_tables.tex")
  )

  # Plots
  plot_arm_target_share(triad_level, output_dir)
  plot_lift_distribution(lift_level, output_dir)
  plot_decoy_share(triad_level, output_dir)

  list(data = all_data, triad_level = triad_level, lift_level = lift_level, summary = summary_tbl, models = mods)
}

# Default run
input_files <- c(
  "data/decoy_preprocessed_gpt41mini.csv",
  "data/decoy_preprocessed_gpt52.csv"
)
model_labels <- c("GPT-4.1 Mini", "GPT-5-2")

run_decoy_analysis(input_files, model_labels, output_dir = "output/decoy/")
