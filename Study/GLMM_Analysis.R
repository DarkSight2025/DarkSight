# MobileViews RQ2: Core GLMM Analysis (publication-ready)
#
# Computes (prints only)
# - HUMANS (llm==0): GLMMs for recall (GT+) and false positive (GT-) with hint × prefilter
# - Arm-level macro/micro metrics (recall, FPR, precision, F1) adjusted by label prevalence
# - Likelihood-ratio tests for interaction terms
# - AMEs for hint within each prefilter level (probability scale)
# - Diagnostics: singularity flags, Pearson overdispersion, ICCs
# - Key fixed-effects with Wald CIs

# Inputs (CSV files in DATA_DIR):
#   recall_long.csv, fp_long.csv, pattern_size.csv
#


suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr)
  library(lme4);  library(emmeans)
})

# -------- paths --------
DATA_DIR <- "annotations_anaylsis"
recall <- read_csv(file.path(DATA_DIR, "recall_long.csv"), show_col_types = FALSE)
fp     <- read_csv(file.path(DATA_DIR, "fp_long.csv"),     show_col_types = FALSE)
psize  <- suppressWarnings(read_csv(file.path(DATA_DIR, "pattern_size.csv"), show_col_types = FALSE))

# -------- helpers --------
fac01 <- function(x, lvl0="False", lvl1="True") factor(ifelse(as.integer(x)==1, lvl1, lvl0), levels=c(lvl0,lvl1))
wald_ci <- function(est, se, z=1.96) cbind(lo = est - z*se, hi = est + z*se)

# -------- harmonize types & references (RECALL: GT+ rows) --------
recall <- recall %>%
  mutate(
    y         = as.integer(y),
    llm       = as.integer(llm),
    label     = factor(label),
    hint_f    = fac01(hint, "no_hint", "hint"),
    pre_f     = fac01(mixed, "random", "prefiltered"),
    worker_id = factor(worker_id),
    screen_id = factor(screen_id),
    size_rel  = suppressWarnings(as.numeric(size_rel)),
    multi_gt  = factor(as.integer(multi_gt), levels=c(0,1), labels=c("single","multi"))
  )

# Explicit label baseline (prefer Pop-up Ad = 'NG-AD' if present)
base_label <- if ("NG-AD" %in% levels(recall$label)) "NG-AD" else levels(recall$label)[1]
recall$label <- relevel(recall$label, ref = base_label)

# Global z-score for size on positives
size_mean <- mean(recall$size_rel, na.rm = TRUE)
size_sd   <- sd(recall$size_rel,  na.rm = TRUE); if (!is.finite(size_sd) || size_sd==0) size_sd <- 1
recall <- recall %>% mutate(size_z = (size_rel - size_mean)/size_sd)

# -------- harmonize types (FP: GT- rows) --------
fp <- fp %>%
  mutate(
    y         = as.integer(y),
    llm       = as.integer(llm),
    label     = factor(label, levels = levels(recall$label)),
    hint_f    = fac01(hint, "no_hint", "hint"),
    pre_f     = fac01(mixed, "random", "prefiltered"),
    worker_id = factor(worker_id),
    screen_id = factor(screen_id)
  )

# Bring multi_gt to FP (screen-level)
multi_lookup <- recall %>% distinct(screen_id, multi_gt)
fp <- fp %>%
  left_join(multi_lookup, by = "screen_id") %>%
  mutate(multi_gt = tidyr::replace_na(multi_gt, "single") |> factor(levels=c("single","multi")))

# FP rows: size_z = 0; we do NOT include size in FP model
fp$size_z <- 0

# -------- sanity prints --------
cat("Rows (recall)   : HUMANS =", sum(recall$llm==0, na.rm=TRUE), " | LLM =", sum(recall$llm==1, na.rm=TRUE), "\n")
cat("Rows (fp)       : HUMANS =", sum(fp$llm==0,     na.rm=TRUE), " | LLM =", sum(fp$llm==1,     na.rm=TRUE), "\n")
cat("Labels baseline :", base_label, "\n")

# -------- model datasets (HUMANS ONLY for the 2x2 GLMMs) --------
hum_pos <- recall %>% filter(llm == 0, !is.na(size_z)) %>% mutate(hint_f = droplevels(hint_f), pre_f = droplevels(pre_f))
hum_neg <- fp     %>% filter(llm == 0)                    %>% mutate(hint_f = droplevels(hint_f), pre_f = droplevels(pre_f))

# -------- GLMMs (HUMANS): recall (GT+) and false-positives (GT-) --------
fx_rec <- y ~ size_z + label + hint_f + pre_f + hint_f:pre_f + size_z:hint_f + multi_gt +
  (1|worker_id) + (1|screen_id)
m_rec <- glmer(fx_rec, family = binomial, data = hum_pos,
               control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

fx_fp <- y ~ label + hint_f + pre_f + hint_f:pre_f + multi_gt + (1|worker_id) + (1|screen_id)
m_fp  <- glmer(fx_fp,  family = binomial, data = hum_neg,
               control = glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

cat("\n=== MODEL SUMMARIES (HUMANS) ===\n")
print(summary(m_rec))
print(summary(m_fp))

# -------- Likelihood-ratio tests for interaction (hint x prefilter) --------
m_rec_no_int <- update(m_rec, . ~ . - hint_f:pre_f)
m_fp_no_int  <- update(m_fp,  . ~ . - hint_f:pre_f)
cat("\n=== LR tests: interaction hint x prefilter (HUMANS) ===\n")
print(anova(m_rec, m_rec_no_int, test="Chisq"))
print(anova(m_fp,  m_fp_no_int,  test="Chisq"))

# -------- Predicted arm-level rates (probability scale, HUMANS) --------
emm_rec_lab <- emmeans(m_rec, ~ hint_f * pre_f | label, type = "response") %>% as.data.frame()
emm_fp_lab  <- emmeans(m_fp,  ~ hint_f * pre_f | label, type = "response") %>% as.data.frame()
names(emm_rec_lab)[names(emm_rec_lab)=="prob"] <- "recall_hat"
names(emm_fp_lab)[ names(emm_fp_lab) =="prob"] <- "fpr_hat"

# -------- Prevalence for micro averaging --------
screens_all <- union(unique(recall$screen_id), unique(fp$screen_id))
S <- length(screens_all)
if (nrow(psize) > 0) {
  psize <- psize %>% mutate(screen_id = as.character(screen_id),
                            label = factor(label, levels = levels(recall$label)))
  P_table <- psize %>% count(label, name="P_l")
} else {
  P_table <- data.frame(label = levels(recall$label), P_l = 0L)
}
PL <- full_join(P_table, data.frame(label = levels(recall$label)), by="label") %>% mutate(P_l = replace_na(P_l, 0L))
PL <- PL %>% mutate(Nneg_l = pmax(0L, S - P_l))

# Merge predictions with prevalence (HUMANS)
pred_join <- emm_rec_lab %>%
  inner_join(emm_fp_lab, by = c("label","hint_f","pre_f")) %>%
  inner_join(PL, by = "label")

# -------- Macro (equal label weights) --------
macro_metrics <- pred_join %>%
  mutate(
    pi_l        = ifelse(S > 0, P_l / S, 0),
    precision_l = (recall_hat * pi_l) / (recall_hat * pi_l + fpr_hat * (1 - pi_l) + 1e-12),
    F1_l        = 2 * precision_l * recall_hat / (precision_l + recall_hat + 1e-12)
  ) %>%
  group_by(hint_f, pre_f) %>%
  summarise(
    recall_macro    = mean(recall_hat),
    fpr_macro       = mean(fpr_hat),
    precision_macro = mean(precision_l),
    F1_macro        = mean(F1_l),
    .groups = "drop"
  )

# -------- Micro (prevalence-weighted, using expected counts) --------
micro_counts <- pred_join %>%
  group_by(hint_f, pre_f) %>%
  summarise(
    TP   = sum(P_l    * recall_hat),
    Pos  = sum(P_l),
    FP   = sum(Nneg_l * fpr_hat),
    Neg  = sum(Nneg_l),
    .groups = "drop"
  ) %>%
  mutate(
    recall_micro = ifelse(Pos > 0, TP / Pos, NA_real_),
    prec_micro   = ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_),
    F1_micro     = ifelse(is.na(prec_micro) | is.na(recall_micro) | (prec_micro + recall_micro)==0,
                          NA_real_, 2 * prec_micro * recall_micro / (prec_micro + recall_micro))
  )

arm_metrics <- macro_metrics %>%
  inner_join(micro_counts %>% select(hint_f, pre_f, recall_micro, prec_micro, F1_micro),
             by = c("hint_f","pre_f")) %>%
  arrange(pre_f, hint_f)
cat("\n=== ADJUSTED ARM-LEVEL METRICS (HUMANS, standardized) ===\n")
print(arm_metrics)

# -------- Within-arm hint contrasts on prob scale (HUMANS) + CIs --------
emm_rec_hp <- emmeans(m_rec, ~ hint_f | pre_f, type = "response")
emm_fp_hp  <- emmeans(m_fp,  ~ hint_f | pre_f, type = "response")
ame_rec <- contrast(emm_rec_hp, method = "pairwise")
ame_fp  <- contrast(emm_fp_hp,  method = "pairwise")
cat("\n=== AMEs: hint (within prefilter) on recall probability (HUMANS) ===\n")
print(summary(ame_rec, infer = c(TRUE, TRUE), type = "response"))
cat("\n=== AMEs: hint (within prefilter) on FP probability (HUMANS) ===\n")
print(summary(ame_fp, infer = c(TRUE, TRUE), type = "response"))

# -------- Diagnostics (no plots): singularity & Pearson overdispersion --------
cat("\n=== Diagnostics (HUMANS, text-only) ===\n")
cat("Singularity: recall ->", isSingular(m_rec), "; fp ->", isSingular(m_fp), "\n")
overdisp_fun <- function(m) {
  rp <- residuals(m, type = "pearson"); rdf <- df.residual(m)
  chisq <- sum(rp^2); ratio <- chisq / rdf; pval <- pchisq(chisq, df = rdf, lower.tail = FALSE)
  data.frame(pearson_chisq = chisq, df = rdf, ratio = ratio, p = pval)
}
cat("Pearson overdispersion (recall):\n"); print(overdisp_fun(m_rec))
cat("Pearson overdispersion (fp):\n");     print(overdisp_fun(m_fp))

# -------- ICC (variance partition) to justify mixed models --------
vc_rec <- as.data.frame(VarCorr(m_rec)); var_w_rec <- vc_rec$vcov[vc_rec$grp=="worker_id"]; var_s_rec <- vc_rec$vcov[vc_rec$grp=="screen_id"]
if(length(var_w_rec)==0) var_w_rec <- 0; if(length(var_s_rec)==0) var_s_rec <- 0
icc_worker_rec <- var_w_rec / (var_w_rec + var_s_rec + (pi^2/3))
icc_screen_rec <- var_s_rec / (var_w_rec + var_s_rec + (pi^2/3))
cat("\nICC (recall): worker =", round(icc_worker_rec,3), " screen =", round(icc_screen_rec,3), "\n")

vc_fp <- as.data.frame(VarCorr(m_fp)); var_w_fp <- vc_fp$vcov[vc_fp$grp=="worker_id"]; var_s_fp <- vc_fp$vcov[vc_fp$grp=="screen_id"]
if(length(var_w_fp)==0) var_w_fp <- 0; if(length(var_s_fp)==0) var_s_fp <- 0
icc_worker_fp <- var_w_fp / (var_w_fp + var_s_fp + (pi^2/3))
icc_screen_fp <- var_s_fp / (var_w_fp + var_s_fp + (pi^2/3))
cat("ICC (FP):     worker =", round(icc_worker_fp,3), " screen =", round(icc_screen_fp,3), "\n")

# -------- Wald CIs for key fixed effects (quick, reader friendly) --------
rec_cf <- summary(m_rec)$coefficients
key_rec <- rec_cf[rownames(rec_cf) %in% c("size_z","pre_fprefiltered","hint_fhint:pre_fprefiltered","multi_gtmulti"), , drop=FALSE]
key_ci  <- wald_ci(key_rec[,1], key_rec[,2])
cat("\nKey recall fixed-effects (log-odds) with Wald 95% CI:\n")
print(cbind(Estimate = key_rec[,1], SE = key_rec[,2], key_ci))

fp_cf <- summary(m_fp)$coefficients
key_fp <- fp_cf[rownames(fp_cf) %in% c("pre_fprefiltered","hint_fhint","hint_fhint:pre_fprefiltered","multi_gtmulti"), , drop=FALSE]
key_fpci <- wald_ci(key_fp[,1], key_fp[,2])
cat("\nKey FP fixed-effects (log-odds) with Wald 95% CI:\n")
print(cbind(Estimate = key_fp[,1], SE = key_fp[,2], key_fpci))

# -------- Optional: per-label within-arm hint contrasts + BH/FDR adjust --------
do_per_label <- TRUE
if (do_per_label) {
  emm_rec_by <- emmeans(m_rec, ~ hint_f | pre_f * label, type="response")
  perlab_rec <- summary(contrast(emm_rec_by, method="pairwise"), infer=c(TRUE, TRUE))
  pvals <- perlab_rec$p.value
  perlab_rec$padj_BH <- p.adjust(pvals, method="BH")
  cat("\nPer-label within-arm hint effects on recall (BH-adjusted p):\n")
  print(perlab_rec)
}

# -------- Sensitivity: drop outlier workers & refit (robustness) --------
w_stats <- bind_rows(
  hum_pos %>% mutate(set="recall") %>% group_by(worker_id) %>% summarise(n_rec=n(), rec_rate=mean(y), .groups="drop"),
  hum_neg %>% mutate(set="fp")     %>% group_by(worker_id) %>% summarise(n_fp =n(), fp_rate =mean(y), .groups="drop")
) %>%
  group_by(worker_id) %>% summarise(n = sum(coalesce(n_rec,0) + coalesce(n_fp,0)),
                                    fp_rate = max(coalesce(fp_rate,0)), .groups="drop")
fp_cut <- quantile(w_stats$fp_rate, 0.99, na.rm=TRUE)
keep_ids <- w_stats %>% filter(n >= 10, fp_rate <= fp_cut) %>% pull(worker_id)

m_rec_sens <- update(m_rec, data = hum_pos %>% filter(worker_id %in% keep_ids))
m_fp_sens  <- update(m_fp,  data = hum_neg %>% filter(worker_id %in% keep_ids))
cat("\n=== Sensitivity: key coefficients after filtering outlier workers ===\n")
cat("Recall: hint×prefilter (main) ->\n"); print(summary(m_rec)$coefficients["hint_fhint:pre_fprefiltered",])
cat("Recall (sensitivity): hint×prefilter ->\n"); print(summary(m_rec_sens)$coefficients["hint_fhint:pre_fprefiltered",])
cat("FP   : hint×prefilter (main) ->\n"); print(summary(m_fp)$coefficients["hint_fhint:pre_fprefiltered",])
cat("FP   (sensitivity): hint×prefilter ->\n"); print(summary(m_fp_sens)$coefficients["hint_fhint:pre_fprefiltered",])

# -------- OPTIONAL: fallback recall model without screen RE if singular --------
if (isSingular(m_rec)) {
  m_rec_worker <- glmer(update(fx_rec, . ~ . - (1|screen_id)), family=binomial, data=hum_pos,
                        control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
  cat("\nRecall model without screen RE (singularity fallback):\n")
  print(summary(m_rec_worker)$coefficients["hint_fhint:pre_fprefiltered",])
}

# -------- OPTIONAL: LLM-only regressions (recall & FP), for baseline reporting --------
gpt_pos <- recall %>% filter(llm == 1, !is.na(size_z))
gpt_neg <- fp     %>% filter(llm == 1)

if (nrow(gpt_pos) > 0) {
  fx_rec_llm <- y ~ size_z + label + multi_gt + (1|screen_id)
  if (nlevels(factor(gpt_pos$screen_id)) < 2) fx_rec_llm <- as.formula("y ~ size_z + label + multi_gt")
  m_rec_llm <- glmer(fx_rec_llm, family=binomial, data=gpt_pos,
                     control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
  cat("\n=== LLM-only RECALL model summary ===\n"); print(summary(m_rec_llm))
} else {
  cat("\n=== LLM-only RECALL model: no data ===\n")
}

if (nrow(gpt_neg) > 0) {
  fx_fp_llm <- y ~ label + (1|screen_id)  # size_z is 0 on GT- rows
  if (nlevels(factor(gpt_neg$screen_id)) < 2) fx_fp_llm <- as.formula("y ~ label")
  m_fp_llm <- glmer(fx_fp_llm, family=binomial, data=gpt_neg,
                    control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
  cat("\n=== LLM-only FP model summary ===\n"); print(summary(m_fp_llm))
} else {
  cat("\n=== LLM-only FP model: no data ===\n")
}

# -------- NEW: Within–Between SIZE decomposition (Crowd + LLM, recall only) --------
# CROWD (GT+, llm==0)
lab_means <- hum_pos %>% group_by(label) %>%
  summarise(label_mean_size = mean(size_z, na.rm = TRUE), .groups="drop")

hum_pos_wb <- hum_pos %>%
  left_join(lab_means, by="label") %>%
  mutate(size_within = size_z - label_mean_size,
         size_between = label_mean_size)

fx_rec_wb <- y ~ size_within + size_between + hint_f + pre_f + hint_f:pre_f +
  size_within:hint_f + multi_gt + (1|worker_id) + (1|screen_id)

m_rec_wb <- glmer(fx_rec_wb, family=binomial, data=hum_pos_wb,
                  control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

cat("\n--- CROWD recall: within–between size decomposition ---\n")
print(coef(summary(m_rec_wb))[c("size_within","size_between"), , drop=FALSE])

if (isSingular(m_rec_wb)) {
  m_rec_wb_worker <- glmer(update(fx_rec_wb, . ~ . - (1|screen_id)), family=binomial, data=hum_pos_wb,
                           control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
  cat("\n[CROWD WB] Fallback (no screen RE) — size coefficients:\n")
  print(coef(summary(m_rec_wb_worker))[c("size_within","size_between"), , drop=FALSE])
}

# LLM (GT+, llm==1)
if (nrow(gpt_pos) > 0) {
  lab_means_llm <- gpt_pos %>% group_by(label) %>%
    summarise(label_mean_size = mean(size_z, na.rm = TRUE), .groups="drop")
  
  gpt_pos_wb <- gpt_pos %>%
    left_join(lab_means_llm, by="label") %>%
    mutate(size_within = size_z - label_mean_size,
           size_between = label_mean_size)
  
  fx_rec_llm_wb <- y ~ size_within + size_between + multi_gt + (1|screen_id)
  if (nlevels(factor(gpt_pos_wb$screen_id)) < 2) fx_rec_llm_wb <- as.formula("y ~ size_within + size_between + multi_gt")
  
  m_rec_llm_wb <- glmer(fx_rec_llm_wb, family=binomial, data=gpt_pos_wb,
                        control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
  
  cat("\n--- LLM recall: within–between size decomposition ---\n")
  print(coef(summary(m_rec_llm_wb))[c("size_within","size_between"), , drop=FALSE])
  
  if (isSingular(m_rec_llm_wb)) {
    m_rec_llm_wb_ns <- glm(update(fx_rec_llm_wb, . ~ . - (1|screen_id)), family=binomial, data=gpt_pos_wb)
    cat("\n[LLM WB] Fallback (no RE) — size coefficients:\n")
    print(coef(summary(m_rec_llm_wb_ns))[c("size_within","size_between"), , drop=FALSE])
  }
}

# Descriptives across labels (crowd, GT+)
desc_lab <- hum_pos %>% group_by(label) %>%
  summarise(mean_size = mean(size_z, na.rm=TRUE),
            recall    = mean(y), .groups="drop") %>%
  arrange(desc(recall))
cat("\n--- Descriptives across labels (crowd, GT+) ---\n"); print(desc_lab)
