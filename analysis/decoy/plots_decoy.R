plot_arm_target_share <- function(triad_level, output_dir) {
  p <- triad_level %>%
    group_by(model, arm) %>%
    summarise(p_target = mean(p_target, na.rm = TRUE), .groups = "drop") %>%
    ggplot(aes(x = arm, y = p_target, fill = arm)) +
    geom_col(width = 0.65) +
    facet_wrap(~ model) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(x = NULL, y = "Target Share", title = "Target Share by Arm") +
    theme_publication()

  save_plot(p, file.path(output_dir, "target_share_by_arm.pdf"), width = 9, height = 5)
}

plot_lift_distribution <- function(lift_level, output_dir) {
  d <- lift_level %>% filter(is.finite(lift_cond_tc))
  if (nrow(d) == 0) {
    warning("No finite lift_cond_tc values; skipping lift distribution plot.")
    return(invisible(NULL))
  }

  p <- d %>%
    ggplot(aes(x = lift_cond_tc, fill = model)) +
    geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~ model, scales = "free_y") +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(x = "Lift in P(Target | Target vs Competitor)", y = "Triad Count", title = "Decoy Lift Distribution") +
    theme_publication()

  save_plot(p, file.path(output_dir, "lift_distribution_cond_tc.pdf"), width = 10, height = 6)
}

plot_decoy_share <- function(triad_level, output_dir) {
  p <- triad_level %>%
    filter(arm == "treatment") %>%
    group_by(model) %>%
    summarise(p_decoy = mean(p_decoy, na.rm = TRUE), .groups = "drop") %>%
    ggplot(aes(x = model, y = p_decoy, fill = model)) +
    geom_col(width = 0.65) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(x = NULL, y = "Decoy Choice Share", title = "Decoy Share in Treatment Arm") +
    theme_publication() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))

  save_plot(p, file.path(output_dir, "decoy_share_treatment.pdf"), width = 7, height = 5)
}
