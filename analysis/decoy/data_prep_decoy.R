prepare_decoy_data <- function(filepath, model_label = NULL) {
  d <- readr::read_csv(filepath, show_col_types = FALSE)

  if (!"triad_id" %in% names(d) || !"arm" %in% names(d)) {
    stop("Input file must contain triad_id and arm columns.")
  }

  d <- d %>%
    mutate(
      arm = factor(arm, levels = c("baseline", "treatment")),
      model = if (!is.null(model_label)) model_label else model_family,
      chose_target = as.numeric(chose_target),
      chose_competitor = as.numeric(chose_competitor),
      chose_decoy = as.numeric(chose_decoy),
      choice_in_tc = as.logical(choice_in_tc),
      chose_target_cond_tc = as.numeric(chose_target_cond_tc)
    ) %>%
    filter(!is.na(triad_id), arm %in% c("baseline", "treatment"))

  d
}

make_triad_level <- function(d) {
  d %>%
    group_by(model, triad_id, arm) %>%
    summarise(
      p_target = mean(chose_target, na.rm = TRUE),
      p_competitor = mean(chose_competitor, na.rm = TRUE),
      p_decoy = mean(chose_decoy, na.rm = TRUE),
      p_target_cond_tc = mean(chose_target_cond_tc, na.rm = TRUE),
      tc_price_premium_pct = mean(tc_price_premium_pct, na.rm = TRUE),
      tc_rating_adv = mean(tc_rating_adv, na.rm = TRUE),
      dt_price_premium_pct = mean(dt_price_premium_pct, na.rm = TRUE),
      td_rating_adv = mean(td_rating_adv, na.rm = TRUE),
      .groups = "drop"
    )
}

make_lift_level <- function(triad_level) {
  # pivot only arm-dependent outcomes
  wide <- triad_level %>%
    select(model, triad_id, arm, p_target, p_target_cond_tc, p_decoy) %>%
    tidyr::pivot_wider(
      names_from = arm,
      values_from = c(p_target, p_target_cond_tc, p_decoy),
      values_fill = NA
    )

  # keep one row of triad attributes (arm-invariant by construction)
  attrs <- triad_level %>%
    group_by(model, triad_id) %>%
    summarise(
      tc_price_premium_pct = mean(tc_price_premium_pct, na.rm = TRUE),
      tc_rating_adv = mean(tc_rating_adv, na.rm = TRUE),
      dt_price_premium_pct = mean(dt_price_premium_pct, na.rm = TRUE),
      td_rating_adv = mean(td_rating_adv, na.rm = TRUE),
      .groups = "drop"
    )

  wide %>%
    left_join(attrs, by = c("model", "triad_id")) %>%
    mutate(
      lift_raw = p_target_treatment - p_target_baseline,
      lift_cond_tc = p_target_cond_tc_treatment - p_target_cond_tc_baseline
    )
}
