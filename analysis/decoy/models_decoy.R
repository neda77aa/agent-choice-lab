fit_decoy_models <- function(d) {
  # Arm effect on target choice, controlling triad fixed effects
  m_target <- feols(
    chose_target ~ i(arm, ref = "baseline") * model | triad_id,
    data = d,
    vcov = ~ triad_id
  )

  # Conditional target-vs-competitor choice (exclude decoy choices)
  d_tc <- d %>% filter(choice_in_tc)
  m_target_cond_tc <- feols(
    chose_target ~ i(arm, ref = "baseline") * model | triad_id,
    data = d_tc,
    vcov = ~ triad_id
  )

  list(
    m_target = m_target,
    m_target_cond_tc = m_target_cond_tc,
    d_tc = d_tc
  )
}

summarize_lifts <- function(lift_level) {
  lift_level %>%
    group_by(model) %>%
    summarise(
      triads = n(),
      mean_lift_raw = mean(lift_raw, na.rm = TRUE),
      mean_lift_cond_tc = mean(lift_cond_tc, na.rm = TRUE),
      mean_decoy_share_treatment = mean(p_decoy_treatment, na.rm = TRUE),
      .groups = "drop"
    )
}
